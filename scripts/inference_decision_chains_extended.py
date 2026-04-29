"""
Extended decision-chains inference: free generation with probability analysis.

Loads a trained HF checkpoint for the extended (3-5 step) chain model,
runs free generation on unique test vectors (no letter forcing), extracts
softmax distributions at each decision point, and evaluates with a 2^N-way
match metric across all decision-function combos.

Usage:
    python scripts/inference_decision_chains_extended.py
    python scripts/inference_decision_chains_extended.py inference.batch_size=64
"""

import sys
import os
import json
import re
import itertools
import torch
import numpy as np
import hydra
import logging

from collections import defaultdict, Counter
from pathlib import Path
from tqdm import trange
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from hydra.utils import to_absolute_path

# Add parent dir for local imports (transformations, evaluator, trajectory_labeler)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ops.transformations import _format_vec
from evaluation.evaluator_chains_extended import (
    LETTER_TO_FUNC, LETTERS, K,
    parse_chain_output_n, apply_and_trace, extract_input_vector,
)
from evaluation.trajectory_labeler import label_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ── Decision functions (from generate_decision_chains_extended.py) ──

def is_even(x: int) -> int:
    return 1 if x % 2 == 0 else 0


def decision_func_f(L) -> int:
    return is_even(L[0]) * 10 + L[1]


def decision_func_g(L) -> int:
    return is_even(L[3]) * 10 + L[4]


DECISION_FUNCS = {"f": decision_func_f, "g": decision_func_g}
INT_TO_LETTER = {i: chr(ord("a") + i) for i in range(20)}
LETTER_TO_INT = {chr(ord("a") + i): i for i in range(20)}


def extract_output_vector(input_str):
    """Extract the OUTPUT vector from 'INPUT : [ ... ] OUTPUT : [ ... ]'."""
    import re
    match = re.search(r'OUTPUT\s*:\s*\[\s*([\d\s,\-]+)\s*\]', input_str)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None


def resolve_chain(vec, df_choices):
    """Given input vec and a sequence of decision function names ('f'/'g'),
    resolve to a tuple of letters by applying decision funcs at each step."""
    current = list(vec)
    letters = []
    for df_name in df_choices:
        idx = DECISION_FUNCS[df_name](current)
        letter = INT_TO_LETTER[idx]
        letters.append(letter)
        _, func = LETTER_TO_FUNC[letter]
        current = func(list(current), K, [])
    return tuple(letters)


def apply_chain_n(vec, letters):
    """Apply N functions in sequence.
    Returns list of (trace_str, result_vec) for each step."""
    current = list(vec)
    steps = []
    for letter in letters:
        name, func = LETTER_TO_FUNC[letter]
        trace = []
        result = func(list(current), K, trace)
        trace_str = trace[0].replace(name, letter, 1)
        steps.append((trace_str, list(result)))
        current = result
    return steps


# ── Model / data loading ──

def load_model_and_tokenizer(cfg):
    model_dir = Path(to_absolute_path(cfg.inference.modelpath))
    log.info(f"Loading model from {model_dir}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    hf_model.cuda()
    hf_model.eval()

    mem_path = cfg.inference.get("memory_weights", None)
    if mem_path:
        mem_path_abs = to_absolute_path(mem_path)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory_experiment"))
        from exp import MemoryAttention, register_hooks
        ckpt = torch.load(mem_path_abs, map_location="cuda")
        memory = MemoryAttention(ckpt["hidden_dim"], ckpt["n_entries"], ckpt["use_gate"])
        memory.load_state_dict(ckpt["state_dict"])
        memory.to("cuda").to(torch.bfloat16).eval()
        for p in memory.parameters():
            p.requires_grad = False
        register_hooks(hf_model, memory, ckpt["layers"])
        hf_model._memory_module = memory
        log.info(f"Loaded memory from {mem_path_abs} (layers={ckpt['layers']}, "
                 f"n_entries={ckpt['n_entries']}, use_gate={ckpt['use_gate']})")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=to_absolute_path(cfg.data.tokenizer_path)
    )
    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"

    return hf_model, tokenizer


def load_test_data(cfg):
    """Load val.json and metadata.json, deduplicate to unique input strings."""
    test_file = to_absolute_path(cfg.data.test_file)
    with open(test_file) as f:
        raw_data = json.load(f)

    meta_file = os.path.join(os.path.dirname(test_file), "metadata.json")
    with open(meta_file) as f:
        metadata = json.load(f)

    filter_combo = cfg.inference.get("filter_df_combo", None)
    if filter_combo:
        target = list(filter_combo) if isinstance(filter_combo, str) else list(filter_combo)
        before = len(raw_data)
        raw_data = [ex for ex in raw_data if ex.get("decision_funcs") == target]
        log.info(f"Filter df_combo={target}: {before} -> {len(raw_data)} examples")

    # Deduplicate by input string, keeping chain length
    seen = {}
    unique_inputs = []  # list of (input_str, chain_length, input_vec)
    for ex in raw_data:
        input_str = ex["input"]
        if input_str not in seen:
            vec = extract_input_vector(input_str)
            chain_len = len(ex["letters"])
            seen[input_str] = len(unique_inputs)
            unique_inputs.append((input_str, chain_len, vec))

    log.info(f"Loaded {len(raw_data)} examples, {len(unique_inputs)} unique inputs")

    # Count per chain length
    len_counts = Counter(cl for _, cl, _ in unique_inputs)
    for cl in sorted(len_counts):
        log.info(f"  Length {cl}: {len_counts[cl]} unique inputs")

    return unique_inputs, metadata


# ── Ground truth computation ──

def compute_all_gt_traces(vec, chain_length):
    """For a single input vector, compute all 2^N GT traces from decision-function combos."""
    df_names = list(DECISION_FUNCS.keys())
    results = []
    for combo in itertools.product(df_names, repeat=chain_length):
        letters = resolve_chain(vec, combo)
        steps = apply_chain_n(vec, letters)
        output = " ; ".join(s[0] for s in steps)
        results.append({
            "letters": letters,
            "df_combo": combo,
            "output": output,
        })
    return results


# ── Letter token utilities ──

def build_letter_token_map(tokenizer):
    letter_to_tid = {}
    for i in range(20):
        letter = chr(ord("a") + i)
        tids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(tids) == 1, f"Letter '{letter}' tokenizes to {len(tids)} tokens"
        letter_to_tid[letter] = tids[0]
    ordered_tids = [letter_to_tid[chr(ord("a") + i)] for i in range(20)]
    return letter_to_tid, ordered_tids


def tid_to_letter(tid, letter_to_tid):
    for letter, t in letter_to_tid.items():
        if t == tid:
            return letter
    return None


# ── Generation with score extraction ──

def compute_token_entropy(logits):
    """Compute entropy from logits tensor."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def get_top_k_alternatives(logits, tokenizer, k=5):
    """Get top k token alternatives with probabilities."""
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))
    return results


@torch.no_grad()
def generate_with_scores(model, tokenizer, prompts, cfg):
    """
    Batched free generation with per-step logit extraction.

    Returns:
        predictions: list[str] — decoded output after [TRACE]
        per_step_probs: list[list[np.ndarray]] — per example, per step, shape (20,)
        per_step_gen_letters: list[list[str|None]] — per example, per step
        token_level_data: list[list[dict]] — per example, per token, full entropy data
        all_sequences: list[list[int]] — per example, full token sequence (PAD-stripped)
    """
    batch_size = cfg.inference.batch_size
    search_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]
    semicolon_id = tokenizer.encode(";", add_special_tokens=False)[0]
    letter_to_tid, ordered_tids = build_letter_token_map(tokenizer)
    ordered_tids_tensor = torch.tensor(ordered_tids, dtype=torch.long, device="cuda")
    pad_id = tokenizer.pad_token_id

    predictions = []
    per_step_probs = []
    per_step_gen_letters = []
    token_level_data = []  # NEW: full token-level entropy data
    all_sequences = []

    for b in trange(0, len(prompts), batch_size, desc="Generating"):
        batch = prompts[b:b + batch_size]
        batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")

        padded_prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_length=cfg.model.block_size,
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.tolist()
        scores = outputs.scores  # tuple of (batch, vocab_size) tensors

        for i, seq in enumerate(sequences):
            # --- Decode prediction ---
            try:
                split_idx = seq.index(search_token_id)
                end_idx = (seq.index(tokenizer.eos_token_id)
                           if tokenizer.eos_token_id in seq else len(seq))
                pred_text = tokenizer.decode(
                    seq[split_idx + 1:end_idx], skip_special_tokens=True
                ).strip()
            except ValueError:
                pred_text = ""
            predictions.append(pred_text)

            # Collect clean (PAD-stripped) full sequence for embedding extraction
            clean_seq = seq
            while clean_seq and clean_seq[0] == pad_id:
                clean_seq = clean_seq[1:]
            all_sequences.append(clean_seq)

            # --- Find all decision points in generated sequence ---
            gen_part = seq[padded_prompt_len:]
            # Remove EOS/PAD
            eos_id = tokenizer.eos_token_id
            if eos_id in gen_part:
                gen_part = gen_part[:gen_part.index(eos_id)]

            # Decision points: step 0 = index 0, step s = index after s-th semicolon
            decision_points = [0]
            for step_idx, tok in enumerate(gen_part):
                if tok == semicolon_id:
                    decision_points.append(step_idx + 1)

            # Extract probabilities at each decision point (existing logic)
            example_probs = []
            example_letters = []
            for dp in decision_points:
                if dp < len(scores):
                    logits = scores[dp][i].float()
                    full_probs = torch.softmax(logits, dim=-1)
                    letter_probs = full_probs[ordered_tids_tensor].cpu().numpy()
                    example_probs.append(letter_probs)

                    gen_tid = gen_part[dp] if dp < len(gen_part) else None
                    example_letters.append(tid_to_letter(gen_tid, letter_to_tid))
                else:
                    example_probs.append(np.full(20, np.nan))
                    example_letters.append(None)

            per_step_probs.append(example_probs)
            per_step_gen_letters.append(example_letters)

            # --- NEW: Extract full token-level entropy data ---
            example_token_data = []
            for tok_idx, tok_id in enumerate(gen_part):
                if tok_idx < len(scores):
                    logits = scores[tok_idx][i]
                    entropy = compute_token_entropy(logits)
                    probs = torch.softmax(logits.float(), dim=-1)
                    token_prob = probs[tok_id].item()
                    top_k = get_top_k_alternatives(logits, tokenizer, k=5)
                    token_str = tokenizer.decode([tok_id])

                    example_token_data.append({
                        "token": token_str,
                        "token_id": tok_id,
                        "entropy": entropy,
                        "probability": token_prob,
                        "top_k": top_k,
                        "is_decision_point": tok_idx in decision_points,
                    })
                else:
                    token_str = tokenizer.decode([tok_id])
                    example_token_data.append({
                        "token": token_str,
                        "token_id": tok_id,
                        "entropy": 0.0,
                        "probability": 1.0,
                        "top_k": [],
                        "is_decision_point": tok_idx in decision_points,
                    })

            token_level_data.append(example_token_data)

    return predictions, per_step_probs, per_step_gen_letters, token_level_data, all_sequences


# ── Embedding extraction ──

OUTPUT_TOKEN_ID = 62
BRACKET_OPEN_ID = 63
BRACKET_CLOSE_ID = 64
EOS_TOKEN_ID = 91


def find_output_vector_positions(token_ids):
    """Find token positions of the OUTPUT vector [ o1 , o2 , ... ] in a sequence.

    Returns list of positions from '[' to ']' inclusive, or None if not found.
    """
    # Find OUTPUT token
    try:
        output_pos = token_ids.index(OUTPUT_TOKEN_ID)
    except ValueError:
        return None

    # Find first [ after OUTPUT
    open_pos = None
    for j in range(output_pos + 1, len(token_ids)):
        if token_ids[j] == BRACKET_OPEN_ID:
            open_pos = j
            break
    if open_pos is None:
        return None

    # Find matching ]
    close_pos = None
    for j in range(open_pos + 1, len(token_ids)):
        if token_ids[j] == BRACKET_CLOSE_ID:
            close_pos = j
            break
    if close_pos is None:
        return None

    return list(range(open_pos, close_pos + 1))


def find_intermediate_vector_positions(token_ids, prompt_len, embed_idx=0):
    """Find token positions of a selected intermediate result vector in generated trace.

    Searches for [ ... ] bracket pairs in the generated part (after prompt_len).
    embed_idx selects which vector (0-indexed from start). Clips to available range.

    Returns list of positions (absolute indices into token_ids), or None if not found.
    """
    gen_part = token_ids[prompt_len:]
    bracket_pairs = []  # list of (open_abs, close_abs)

    i = 0
    while i < len(gen_part):
        if gen_part[i] == BRACKET_OPEN_ID:
            for j in range(i + 1, len(gen_part)):
                if gen_part[j] == EOS_TOKEN_ID:
                    break
                if gen_part[j] == BRACKET_CLOSE_ID:
                    bracket_pairs.append((prompt_len + i, prompt_len + j))
                    i = j
                    break
        i += 1

    if not bracket_pairs:
        return None

    # Select using embed_idx (0-indexed from start), clip to available range
    idx = max(0, min(embed_idx, len(bracket_pairs) - 1))

    open_pos, close_pos = bracket_pairs[idx]
    return list(range(open_pos, close_pos + 1))


@torch.no_grad()
def extract_last_layer_embeddings(model, tokenizer, prompts, all_sequences, cfg):
    """Extract last-layer hidden states at OUTPUT vector and intermediate vector positions.

    For each example, extracts hidden states from the final transformer layer at:
    - The OUTPUT vector tokens from the prompt (13 tokens for 6-element vectors)
    - A selected intermediate result vector from the generated trace

    Returns:
        embeddings: np.ndarray of shape (N, 2*vec_tokens, n_embd), NaN for missing
        valid_mask: np.ndarray of shape (N,) bool, True if both vectors found
    """
    batch_size = cfg.inference.batch_size
    embed_idx = cfg.inference.embed_intermediate_idx
    n_embd = cfg.model.n_embd

    N = len(all_sequences)

    # Determine vec_token_count from first example's OUTPUT vector
    first_output_pos = find_output_vector_positions(all_sequences[0])
    vec_token_count = len(first_output_pos) if first_output_pos else 13
    total_tokens = vec_token_count * 2

    all_embeddings = np.full((N, total_tokens, n_embd), np.nan, dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for b in trange(0, N, batch_size, desc="Extracting embeddings"):
        batch_seqs = all_sequences[b:b + batch_size]
        batch_prompts = prompts[b:b + batch_size]
        bs = len(batch_seqs)

        # Left-pad sequences for batched forward pass
        max_len = max(len(seq) for seq in batch_seqs)
        pad_id = tokenizer.pad_token_id

        padded_ids = []
        attention_masks = []
        pad_offsets = []
        for seq in batch_seqs:
            pad_len = max_len - len(seq)
            padded_ids.append([pad_id] * pad_len + seq)
            attention_masks.append([0] * pad_len + [1] * len(seq))
            pad_offsets.append(pad_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device="cuda")
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device="cuda")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (bs, max_len, n_embd)

        for i in range(bs):
            seq = batch_seqs[i]
            prompt_len = len(batch_prompts[i])
            pad_offset = pad_offsets[i]

            output_pos = find_output_vector_positions(seq)
            intermediate_pos = find_intermediate_vector_positions(seq, prompt_len, embed_idx)

            if output_pos is None or intermediate_pos is None:
                continue
            if len(output_pos) != vec_token_count or len(intermediate_pos) != vec_token_count:
                continue

            # Extract OUTPUT vector embeddings
            for j, pos in enumerate(output_pos):
                all_embeddings[b + i, j, :] = (
                    last_hidden[i, pos + pad_offset, :].float().cpu().numpy()
                )

            # Extract intermediate vector embeddings
            for j, pos in enumerate(intermediate_pos):
                all_embeddings[b + i, vec_token_count + j, :] = (
                    last_hidden[i, pos + pad_offset, :].float().cpu().numpy()
                )

            valid_mask[b + i] = True

        del outputs, last_hidden
        torch.cuda.empty_cache()

    return all_embeddings, valid_mask


# ── Metrics ──

def compute_metrics(unique_inputs, predictions, per_step_probs, per_step_gen_letters, metadata, token_level_data=None):
    """Compute all metrics: N-way match, block validity, probabilities, tuple classification."""

    # Build train/test tuple sets per length
    train_tuples_per_len = {}
    test_tuples_per_len = {}
    for length_str, tuples in metadata["train_tuples_per_length"].items():
        train_tuples_per_len[int(length_str)] = set(tuple(t) for t in tuples)
    for length_str, tuples in metadata["test_tuples_per_length"].items():
        test_tuples_per_len[int(length_str)] = set(tuple(t) for t in tuples)

    n = len(unique_inputs)

    # Overall counters
    n_way_correct = 0
    parseable_count = 0
    all_blocks_valid_count = 0

    # Valid chain metrics
    # Valid chain = correct chain length AND all steps chose f/g letter AND all blocks valid
    valid_chain_count = 0
    # Valid solution = valid chain AND final vector matches prompt OUTPUT
    valid_solution_count = 0

    # All-steps-valid metrics (lenient: ignores chain length, only checks steps are f/g and valid)
    all_steps_valid_count = 0

    # Per chain length
    n_way_per_len = defaultdict(lambda: {"correct": 0, "total": 0})
    parseable_per_len = defaultdict(lambda: {"count": 0, "total": 0})
    valid_chain_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    valid_solution_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    all_steps_valid_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    # Per-step block validity
    per_step_valid = defaultdict(int)
    per_step_total = defaultdict(int)

    # Tuple classification
    chosen_train = 0
    chosen_test = 0
    chosen_neither = 0
    chosen_train_per_len = defaultdict(int)
    chosen_test_per_len = defaultdict(int)

    # Per-step probability accumulators
    prob_f_per_step = defaultdict(list)
    prob_g_per_step = defaultdict(list)
    decision_mass_per_step = defaultdict(list)

    # Per-function accuracy per step
    per_func_per_step = defaultdict(lambda: {"correct": 0, "total": 0})

    # Tuple frequency
    tuple_frequency = Counter()
    tuple_frequency_per_len = defaultdict(Counter)

    per_example_results = []

    # Entropy monotonicity tracking
    entropy_monotone_count = 0
    entropy_violation_counts = []  # per-example violation count

    # For visualization
    vis_per_step_probs = []
    vis_per_step_idx_f = []
    vis_per_step_idx_g = []
    vis_gen_letters = []
    vis_vecs = []
    vis_is_valid_chain = []
    vis_is_valid_solution = []
    vis_has_different_final = []
    vis_is_entropy_monotone = []
    vis_entropy_violation_count = []
    vis_per_step_entropy = []

    for i, (input_str, expected_chain_len, input_vec) in enumerate(unique_inputs):
        pred = predictions[i].strip()
        step_probs = per_step_probs[i]
        gen_letters = per_step_gen_letters[i]

        n_way_per_len[expected_chain_len]["total"] += 1
        parseable_per_len[expected_chain_len]["total"] += 1

        # Compute all 2^N GT traces (all 2^n combinations of f/g at each step)
        gt_traces = compute_all_gt_traces(input_vec, expected_chain_len)

        # N-way match - compare parsed letter sequences AND verify computation
        # Both must be correct for N-way match
        pred_blocks_for_nway = parse_chain_output_n(pred)
        pred_letters = tuple(b["letter"] for b in pred_blocks_for_nway if b is not None) if pred_blocks_for_nway else ()

        matched_gt = None
        for gt in gt_traces:
            if pred_letters == tuple(gt["letters"]):
                # Letters match - now verify computation by comparing result vectors
                gt_blocks = parse_chain_output_n(gt["output"])
                if pred_blocks_for_nway and gt_blocks and len(pred_blocks_for_nway) == len(gt_blocks):
                    all_vecs_match = True
                    for pb, gb in zip(pred_blocks_for_nway, gt_blocks):
                        if pb is None or gb is None or pb.get("vec") != gb.get("vec"):
                            all_vecs_match = False
                            break
                    if all_vecs_match:
                        matched_gt = gt
                        break

        is_correct = matched_gt is not None
        if is_correct:
            n_way_correct += 1
            n_way_per_len[expected_chain_len]["correct"] += 1

        # Parse prediction into blocks
        pred_blocks = parse_chain_output_n(pred)
        is_parseable = pred_blocks is not None and len(pred_blocks) >= 1
        if is_parseable:
            parseable_count += 1
            parseable_per_len[expected_chain_len]["count"] += 1

        # Per-step block validity + function accuracy
        current_vec = list(input_vec)
        n_pred_steps = len(pred_blocks) if pred_blocks else 0
        steps_valid = []

        for s in range(n_pred_steps):
            block = pred_blocks[s]
            per_step_total[s] += 1

            if block is None or current_vec is None:
                steps_valid.append(False)
                current_vec = None
                continue

            letter = block["letter"]
            expected_result, expected_block = apply_and_trace(letter, current_vec)
            block_valid = (expected_block is not None
                           and block["block"] == expected_block)

            steps_valid.append(block_valid)
            if block_valid:
                per_step_valid[s] += 1

            per_func_per_step[(s, letter)]["total"] += 1
            if block_valid:
                per_func_per_step[(s, letter)]["correct"] += 1

            # Advance vector
            if expected_result is not None and block_valid:
                current_vec = expected_result
            elif block["vec"] is not None:
                current_vec = block["vec"]
            else:
                current_vec = None

        if steps_valid and all(steps_valid):
            all_blocks_valid_count += 1

        # Check if each step chose a letter from f or g decision functions
        # AND all blocks are valid => "valid chain"
        valid_chain_per_len[expected_chain_len]["total"] += 1
        valid_solution_per_len[expected_chain_len]["total"] += 1
        steps_chose_fg = []
        check_vec = list(input_vec)
        final_computed_vec = None

        for s in range(n_pred_steps):
            if pred_blocks and s < len(pred_blocks) and pred_blocks[s] is not None:
                chosen_letter = pred_blocks[s]["letter"]
                if check_vec is not None and len(check_vec) >= 5:
                    # Get correct letters from decision functions
                    idx_f = decision_func_f(check_vec)
                    idx_g = decision_func_g(check_vec)
                    letter_f = INT_TO_LETTER[idx_f]
                    letter_g = INT_TO_LETTER[idx_g]

                    # Check if chosen letter is f or g
                    chose_fg = (chosen_letter == letter_f or chosen_letter == letter_g)
                    steps_chose_fg.append(chose_fg)

                    # Advance check_vec for next step
                    entry = LETTER_TO_FUNC.get(chosen_letter)
                    if entry is not None:
                        _, func = entry
                        check_vec = func(list(check_vec), K, [])
                        final_computed_vec = check_vec
                    else:
                        check_vec = None
                else:
                    steps_chose_fg.append(False)
                    check_vec = None
            else:
                steps_chose_fg.append(False)

        # All-steps-valid = all steps chose f/g AND all blocks valid (ignores chain length)
        all_chose_fg = len(steps_chose_fg) == n_pred_steps and all(steps_chose_fg) if steps_chose_fg else False
        all_valid = steps_valid and all(steps_valid) if steps_valid else False
        is_all_steps_valid = all_chose_fg and all_valid and n_pred_steps > 0

        # Valid chain = correct chain length AND all steps chose f/g AND all blocks valid
        correct_length = n_pred_steps == expected_chain_len
        is_valid_chain = correct_length and is_all_steps_valid
        # Valid solution = valid chain AND final vector matches prompt OUTPUT
        is_valid_solution = False
        has_different_final = False

        if is_all_steps_valid:
            all_steps_valid_count += 1
            all_steps_valid_per_len[expected_chain_len]["valid"] += 1
        all_steps_valid_per_len[expected_chain_len]["total"] += 1

        if is_valid_chain:
            valid_chain_count += 1
            valid_chain_per_len[expected_chain_len]["valid"] += 1

            # Check if final vector matches OUTPUT vector in prompt
            output_vec = extract_output_vector(input_str)
            if output_vec is not None and final_computed_vec is not None:
                if list(final_computed_vec) == list(output_vec):
                    is_valid_solution = True
                    valid_solution_count += 1
                    valid_solution_per_len[expected_chain_len]["valid"] += 1
                else:
                    has_different_final = True

        # Tuple classification
        chosen_tuple = tuple(b["letter"] for b in pred_blocks if b is not None) if pred_blocks else ()
        if len(chosen_tuple) == n_pred_steps and n_pred_steps > 0:
            tuple_frequency[chosen_tuple] += 1
            tuple_frequency_per_len[n_pred_steps][chosen_tuple] += 1

            train_set = train_tuples_per_len.get(n_pred_steps, set())
            test_set = test_tuples_per_len.get(n_pred_steps, set())
            if chosen_tuple in train_set:
                chosen_train += 1
                chosen_train_per_len[n_pred_steps] += 1
            elif chosen_tuple in test_set:
                chosen_test += 1
                chosen_test_per_len[n_pred_steps] += 1
            else:
                chosen_neither += 1

        # Per-step probability metrics
        current_vec_for_probs = list(input_vec)
        step_idx_f_list = []
        step_idx_g_list = []

        for s in range(len(step_probs)):
            probs = step_probs[s]
            if current_vec_for_probs is not None and len(current_vec_for_probs) >= 5:
                idx_f = decision_func_f(current_vec_for_probs)
                idx_g = decision_func_g(current_vec_for_probs)
                step_idx_f_list.append(idx_f)
                step_idx_g_list.append(idx_g)

                if not np.isnan(probs).any():
                    p_f = float(probs[idx_f])
                    p_g = float(probs[idx_g])
                    prob_f_per_step[s].append(p_f)
                    prob_g_per_step[s].append(p_g)
                    if idx_f == idx_g:
                        decision_mass_per_step[s].append(p_f)
                    else:
                        decision_mass_per_step[s].append(p_f + p_g)

                # Advance vector using model's actual chosen letter
                gen_l = gen_letters[s] if s < len(gen_letters) else None
                if gen_l is not None:
                    entry = LETTER_TO_FUNC.get(gen_l)
                    if entry is not None:
                        _, func = entry
                        current_vec_for_probs = func(list(current_vec_for_probs), K, [])
                    else:
                        current_vec_for_probs = None
                else:
                    current_vec_for_probs = None
            else:
                step_idx_f_list.append(None)
                step_idx_g_list.append(None)
                current_vec_for_probs = None

        # Entropy monotonicity: compute Shannon entropy at each decision point
        # and check if it decreases monotonically
        step_entropies = []
        for s in range(len(step_probs)):
            probs = step_probs[s]
            if not np.isnan(probs).any():
                # Shannon entropy in nats: H = -sum(p * log(p)) for p > 0
                p = probs[probs > 0]
                h = -float(np.sum(p * np.log(p)))
                step_entropies.append(h)
            else:
                step_entropies.append(None)

        # Check monotonicity (entropy should decrease at every step)
        epsilon = 0.01  # tolerance for negligible increases
        violations = 0
        is_monotone = True
        if len(step_entropies) >= 2:
            for k in range(len(step_entropies) - 1):
                if step_entropies[k] is not None and step_entropies[k+1] is not None:
                    if step_entropies[k+1] > step_entropies[k] + epsilon:
                        violations += 1
                        is_monotone = False
        else:
            # Single step or no steps — trivially monotone
            is_monotone = True

        if is_monotone:
            entropy_monotone_count += 1
        entropy_violation_counts.append(violations)

        # Collect visualization data
        vis_vecs.append(list(input_vec))
        vis_per_step_probs.append([p.tolist() if hasattr(p, 'tolist') else list(p) for p in step_probs])
        vis_per_step_idx_f.append(step_idx_f_list)
        vis_per_step_idx_g.append(step_idx_g_list)
        vis_gen_letters.append(list(gen_letters))
        vis_is_valid_chain.append(is_valid_chain)
        vis_is_valid_solution.append(is_valid_solution)
        vis_has_different_final.append(has_different_final)
        vis_is_entropy_monotone.append(is_monotone)
        vis_entropy_violation_count.append(violations)
        vis_per_step_entropy.append(step_entropies)

        # Per-example row
        if len(per_example_results) < 200:
            row = {
                "input": input_str,
                "chain_length": expected_chain_len,
                "prediction": pred,
                "gen_letters": list(gen_letters),
                "n_way_match": is_correct,
                "matched_combo": (",".join(matched_gt["df_combo"])) if matched_gt else None,
                "parseable": is_parseable,
                "n_pred_steps": n_pred_steps,
                "all_blocks_valid": all(steps_valid) if steps_valid else False,
                "per_step_valid": steps_valid,
                "chosen_tuple": ",".join(chosen_tuple) if chosen_tuple else None,
            }
            # Add per-step probabilities
            for s in range(len(step_probs)):
                if s < len(step_idx_f_list) and step_idx_f_list[s] is not None:
                    probs = step_probs[s]
                    if not np.isnan(probs).any():
                        row[f"step{s+1}_P_f"] = float(probs[step_idx_f_list[s]])
                        row[f"step{s+1}_P_g"] = float(probs[step_idx_g_list[s]])
            per_example_results.append(row)

    # ── Aggregate metrics ──
    metrics = {
        "n_way_match": n_way_correct / max(n, 1),
        "all_blocks_valid": all_blocks_valid_count / max(parseable_count, 1),
        "parseable_fraction": parseable_count / max(n, 1),

        "chosen_train_tuple_frac": chosen_train / max(parseable_count, 1),
        "chosen_test_tuple_frac": chosen_test / max(parseable_count, 1),
        "chosen_neither_frac": chosen_neither / max(parseable_count, 1),

        # Valid chain = correct length + all f/g + all blocks valid (output may differ)
        "valid_chain_frac": valid_chain_count / max(n, 1),
        "num_valid_chains": valid_chain_count,

        # Valid solution = valid chain + final vector matches prompt OUTPUT
        "valid_solution_frac": valid_solution_count / max(n, 1),
        "num_valid_solutions": valid_solution_count,

        # All-steps-valid metrics (lenient: any chain length, all f/g, all blocks valid)
        "all_steps_valid_frac": all_steps_valid_count / max(n, 1),
        "num_all_steps_valid": all_steps_valid_count,

        "num_unique_inputs": n,
        "num_parseable": parseable_count,
        "num_n_way_correct": n_way_correct,

        # Entropy monotonicity metrics
        "entropy_monotone_frac": entropy_monotone_count / max(n, 1),
        "num_entropy_monotone": entropy_monotone_count,
        "num_entropy_non_monotone": n - entropy_monotone_count,
        "avg_entropy_violations": float(np.mean(entropy_violation_counts)) if entropy_violation_counts else 0.0,
    }

    # Entropy monotonicity accuracy correlation
    # Accuracy among monotone vs non-monotone chains
    mono_correct = sum(1 for i in range(n)
                       if vis_is_entropy_monotone[i] and vis_is_valid_solution[i])
    mono_total = sum(1 for i in range(n) if vis_is_entropy_monotone[i])
    nonmono_correct = sum(1 for i in range(n)
                          if not vis_is_entropy_monotone[i] and vis_is_valid_solution[i])
    nonmono_total = sum(1 for i in range(n) if not vis_is_entropy_monotone[i])
    if mono_total > 0:
        metrics["entropy_monotone_accuracy"] = mono_correct / mono_total
    if nonmono_total > 0:
        metrics["entropy_non_monotone_accuracy"] = nonmono_correct / nonmono_total
    if mono_total > 0 and nonmono_total > 0:
        metrics["entropy_monotonicity_gap"] = (mono_correct / mono_total) - (nonmono_correct / nonmono_total)

    # Per chain length
    for cl in sorted(n_way_per_len):
        d = n_way_per_len[cl]
        metrics[f"n_way_match_len{cl}"] = d["correct"] / max(d["total"], 1)
        metrics[f"n_len{cl}"] = d["total"]

        pd = parseable_per_len[cl]
        metrics[f"parseable_frac_len{cl}"] = pd["count"] / max(pd["total"], 1)

        train_n = chosen_train_per_len.get(cl, 0)
        test_n = chosen_test_per_len.get(cl, 0)
        total_cl = train_n + test_n
        if total_cl > 0:
            metrics[f"chosen_train_frac_len{cl}"] = train_n / total_cl
            metrics[f"chosen_test_frac_len{cl}"] = test_n / total_cl

        # Valid chain / valid solution per length
        vc = valid_chain_per_len[cl]
        if vc["total"] > 0:
            metrics[f"valid_chain_frac_len{cl}"] = vc["valid"] / vc["total"]
        vs = valid_solution_per_len[cl]
        if vs["total"] > 0:
            metrics[f"valid_solution_frac_len{cl}"] = vs["valid"] / vs["total"]

        # All-steps-valid per length
        asv = all_steps_valid_per_len[cl]
        if asv["total"] > 0:
            metrics[f"all_steps_valid_frac_len{cl}"] = asv["valid"] / asv["total"]

    # Per-step block validity
    for s in range(5):
        if per_step_total[s] > 0:
            metrics[f"step{s+1}_block_valid"] = per_step_valid[s] / per_step_total[s]

    # Per-step probability metrics
    all_prob_f = []
    all_prob_g = []
    for s in range(5):
        if prob_f_per_step[s]:
            avg_f = float(np.mean(prob_f_per_step[s]))
            avg_g = float(np.mean(prob_g_per_step[s]))
            avg_mass = float(np.mean(decision_mass_per_step[s]))
            metrics[f"step{s+1}_avg_prob_f"] = avg_f
            metrics[f"step{s+1}_avg_prob_g"] = avg_g
            metrics[f"step{s+1}_decision_mass"] = avg_mass
            all_prob_f.extend(prob_f_per_step[s])
            all_prob_g.extend(prob_g_per_step[s])

    if all_prob_f:
        metrics["avg_prob_f"] = float(np.mean(all_prob_f))
    if all_prob_g:
        metrics["avg_prob_g"] = float(np.mean(all_prob_g))

    # Top tuples per length
    tuple_dist = {}
    for cl in sorted(tuple_frequency_per_len):
        top = tuple_frequency_per_len[cl].most_common(20)
        tuple_dist[str(cl)] = {",".join(t): c for t, c in top}
    metrics["tuple_distribution_per_length"] = tuple_dist

    # Visualization data bundle
    vis_data = {
        "per_step_probs": vis_per_step_probs,
        "vecs": vis_vecs,
        "per_step_idx_f": vis_per_step_idx_f,
        "per_step_idx_g": vis_per_step_idx_g,
        "gen_letters": vis_gen_letters,
        "token_level_data": token_level_data,  # NEW: full token entropy data
        "is_valid_chain": vis_is_valid_chain,
        "is_valid_solution": vis_is_valid_solution,
        "has_different_final": vis_has_different_final,
        "is_entropy_monotone": vis_is_entropy_monotone,
        "entropy_violation_count": vis_entropy_violation_count,
        "per_step_entropy": vis_per_step_entropy,
    }

    return metrics, per_example_results, vis_data


# ── Main ──

@hydra.main(config_path="../config", config_name="base_decision_chains_extended", version_base=None)
def main(cfg: DictConfig):
    log.info("Extended decision-chains inference: free generation + probability analysis")

    # 1. Load
    model, tokenizer = load_model_and_tokenizer(cfg)
    unique_inputs, metadata = load_test_data(cfg)

    if cfg.inference.sampling.sample_test_set:
        n = int(cfg.inference.sampling.num_test)
        unique_inputs = unique_inputs[:n]
        log.info(f"Subsampled to {len(unique_inputs)} inputs")

    log.info(f"Running inference on {len(unique_inputs)} unique inputs")

    # 2. Build prompts
    prompts = []
    for input_str, _, _ in unique_inputs:
        text = f"{tokenizer.bos_token} {input_str} {cfg.data.split_str}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append(ids)

    # 3. Generate
    predictions, step_probs, step_gen_letters, token_level_data, all_sequences = generate_with_scores(
        model, tokenizer, prompts, cfg
    )

    # 3b. Extract last-layer embeddings (must happen before del model)
    log.info("Extracting last-layer embeddings...")
    embeddings, embed_valid_mask = extract_last_layer_embeddings(
        model, tokenizer, prompts, all_sequences, cfg
    )
    log.info(f"Embeddings extracted: shape {embeddings.shape}, "
             f"valid: {embed_valid_mask.sum()}/{len(embed_valid_mask)}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # 4. Metrics
    metrics, per_example_results, vis_data = compute_metrics(
        unique_inputs, predictions, step_probs, step_gen_letters, metadata,
        token_level_data=token_level_data
    )

    # 5. Print summary
    print("\n" + "=" * 60)
    print("  Extended Decision-Chains Inference Results")
    print("=" * 60)

    print(f"\nOverall ({metrics['num_unique_inputs']} unique inputs):")
    print(f"  {'n_way_match':35s}: {metrics['n_way_match']:.4f}")
    print(f"  {'all_blocks_valid':35s}: {metrics['all_blocks_valid']:.4f}")
    print(f"  {'parseable_fraction':35s}: {metrics['parseable_fraction']:.4f}")

    print(f"\nPer chain length:")
    for cl in [3, 4, 5]:
        key = f"n_way_match_len{cl}"
        if key in metrics:
            n_key = f"n_len{cl}"
            print(f"  Length {cl} (n={metrics[n_key]}):")
            print(f"    {'n_way_match':31s}: {metrics[key]:.4f}")
            pf = metrics.get(f"parseable_frac_len{cl}", 0)
            print(f"    {'parseable_fraction':31s}: {pf:.4f}")
            tf = metrics.get(f"chosen_train_frac_len{cl}")
            if tf is not None:
                print(f"    {'chosen_train_frac':31s}: {tf:.4f}")
                print(f"    {'chosen_test_frac':31s}: {metrics.get(f'chosen_test_frac_len{cl}', 0):.4f}")

    print(f"\nTuple origin (overall):")
    print(f"  {'chosen_train_tuple_frac':35s}: {metrics['chosen_train_tuple_frac']:.4f}")
    print(f"  {'chosen_test_tuple_frac':35s}: {metrics['chosen_test_tuple_frac']:.4f}")
    print(f"  {'chosen_neither_frac':35s}: {metrics['chosen_neither_frac']:.4f}")

    print(f"\nPer-step block validity:")
    for s in range(5):
        key = f"step{s+1}_block_valid"
        if key in metrics:
            print(f"  {key:35s}: {metrics[key]:.4f}")

    print(f"\nPer-step decision-point probabilities:")
    for s in range(5):
        f_key = f"step{s+1}_avg_prob_f"
        if f_key in metrics:
            print(f"  Step {s+1}: P(f)={metrics[f_key]:.4f}  "
                  f"P(g)={metrics[f'step{s+1}_avg_prob_g']:.4f}  "
                  f"mass={metrics[f'step{s+1}_decision_mass']:.4f}")

    if "avg_prob_f" in metrics:
        print(f"\n  Grand avg P(f)={metrics['avg_prob_f']:.4f}  "
              f"P(g)={metrics['avg_prob_g']:.4f}")

    # Top tuples per length
    tuple_dist = metrics.get("tuple_distribution_per_length", {})
    for cl_str in sorted(tuple_dist):
        print(f"\n  Top tuples (length {cl_str}):")
        train_set = set(
            tuple(t) for t in metadata["train_tuples_per_length"].get(cl_str, [])
        )
        for tup_str, count in list(tuple_dist[cl_str].items())[:5]:
            tup = tuple(tup_str.split(","))
            label = "TRAIN" if tup in train_set else "TEST"
            print(f"    ({tup_str}): {count:5d}  [{label}]")

    print(f"\nValid chain metrics (correct length + all f/g + all blocks valid):")
    print(f"  {'valid_chain_frac':35s}: {metrics['valid_chain_frac']:.4f}")
    print(f"  {'num_valid_chains':35s}: {metrics['num_valid_chains']}")

    print(f"\nValid solution metrics (valid chain + output matches):")
    print(f"  {'valid_solution_frac':35s}: {metrics['valid_solution_frac']:.4f}")
    print(f"  {'num_valid_solutions':35s}: {metrics['num_valid_solutions']}")

    print(f"\nAll-steps-valid metrics (any chain length, all steps f/g + valid):")
    print(f"  {'all_steps_valid_frac':35s}: {metrics['all_steps_valid_frac']:.4f}")
    print(f"  {'num_all_steps_valid':35s}: {metrics['num_all_steps_valid']}")

    print(f"\nEntropy monotonicity (decision-point entropy decreases at every step):")
    print(f"  {'entropy_monotone_frac':35s}: {metrics['entropy_monotone_frac']:.4f}")
    print(f"  {'num_entropy_monotone':35s}: {metrics['num_entropy_monotone']}")
    print(f"  {'num_entropy_non_monotone':35s}: {metrics['num_entropy_non_monotone']}")
    print(f"  {'avg_entropy_violations':35s}: {metrics['avg_entropy_violations']:.4f}")
    if "entropy_monotone_accuracy" in metrics:
        print(f"  {'monotone_accuracy (valid sol.)':35s}: {metrics['entropy_monotone_accuracy']:.4f}")
    if "entropy_non_monotone_accuracy" in metrics:
        print(f"  {'non_monotone_accuracy (valid sol.)':35s}: {metrics['entropy_non_monotone_accuracy']:.4f}")
    if "entropy_monotonicity_gap" in metrics:
        print(f"  {'gap (mono - non-mono)':35s}: {metrics['entropy_monotonicity_gap']:+.4f}")

    print(f"\nTotal: {metrics['num_unique_inputs']} inputs, "
          f"{metrics['num_n_way_correct']} correct ({metrics['n_way_match']:.2%})")

    # 6. Save
    output_dir = Path(to_absolute_path(cfg.eval.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": {k: v for k, v in metrics.items()
                    if k != "tuple_distribution_per_length"},
        "tuple_distribution_per_length": tuple_dist,
        "per_example_results": per_example_results,
    }
    results_path = output_dir / "inference_results_extended.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save raw probability arrays for downstream analysis
    # Use object arrays since chain lengths vary
    np.savez(
        output_dir / "decision_point_probs_extended.npz",
        per_step_probs=np.array(vis_data["per_step_probs"], dtype=object),
        unique_vecs=np.array(vis_data["vecs"]),
        per_step_idx_f=np.array(vis_data["per_step_idx_f"], dtype=object),
        per_step_idx_g=np.array(vis_data["per_step_idx_g"], dtype=object),
        gen_letters_all=np.array(vis_data["gen_letters"], dtype=object),
    )
    log.info(f"Probability arrays saved to {output_dir / 'decision_point_probs_extended.npz'}")

    # 7. Label trajectories as correct/incorrect
    labeled_path = output_dir / "labeled_dataset.jsonl"
    label_trajectories(unique_inputs, predictions, str(labeled_path))

    # 7b. Save labeled embeddings NPZ
    valid_solution_labels = np.array(vis_data["is_valid_solution"], dtype=bool)
    chain_lengths = np.array([cl for _, cl, _ in unique_inputs], dtype=np.int32)
    input_vecs = np.array([vec for _, _, vec in unique_inputs], dtype=np.int32)

    npz_path = output_dir / "labeled_dataset_pure.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        labels=valid_solution_labels,
        chain_lengths=chain_lengths,
        input_vecs=input_vecs,
        embed_valid_mask=embed_valid_mask,
    )
    log.info(f"Labeled embeddings saved to {npz_path} "
             f"({embeddings.nbytes / 1024**2:.1f} MB)")

    # 8. Interactive HTML visualization
    from visualization.visualize_superposition import generate_html_extended

    html_limit = getattr(cfg.inference, "html_samples", 0)
    if html_limit and html_limit < len(unique_inputs):
        log.info(f"Limiting HTML visualization to {html_limit} / {len(unique_inputs)} examples")
        h = html_limit
        html_vis_data = {k: v[:h] if isinstance(v, list) else v for k, v in vis_data.items()}
        html_unique_inputs = unique_inputs[:h]
        html_predictions = predictions[:h]
    else:
        html_vis_data = vis_data
        html_unique_inputs = unique_inputs
        html_predictions = predictions

    # Build lightweight mono data from ALL examples (not html-limited)
    mono_data_all = []
    for i, (input_str, chain_len, input_vec) in enumerate(unique_inputs):
        mono_data_all.append({
            "per_step_entropy": vis_data["per_step_entropy"][i],
            "is_entropy_monotone": vis_data["is_entropy_monotone"][i],
            "chain_len": chain_len,
            "entropy_violation_count": vis_data["entropy_violation_count"][i],
        })

    html_dir = output_dir / "html_view"
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / "letter_probs_extended.html"
    generate_html_extended(
        html_vis_data["per_step_probs"],
        html_vis_data["vecs"],
        str(html_path),
        per_step_idx_f=html_vis_data["per_step_idx_f"],
        per_step_idx_g=html_vis_data["per_step_idx_g"],
        gen_letters_all=html_vis_data["gen_letters"],
        train_tuples_per_length=metadata["train_tuples_per_length"],
        token_level_data=html_vis_data.get("token_level_data"),
        unique_inputs=html_unique_inputs,
        predictions=html_predictions,
        metrics=metrics,
        is_valid_chain=html_vis_data.get("is_valid_chain"),
        is_valid_solution=html_vis_data.get("is_valid_solution"),
        has_different_final=html_vis_data.get("has_different_final"),
        is_entropy_monotone=html_vis_data.get("is_entropy_monotone"),
        entropy_violation_count=html_vis_data.get("entropy_violation_count"),
        per_step_entropy=html_vis_data.get("per_step_entropy"),
        mono_data_all=mono_data_all,
    )


if __name__ == "__main__":
    main()
