"""
STOP token experiment inference: free generation with STOP analysis.

Extends inference_decision_chains_extended.py with:
1. STOP detection in free generation
2. STOP probability probing at wrong-branch and correct-branch positions

Usage:
    python scripts/inference_decision_chains_stop.py
    python scripts/inference_decision_chains_stop.py inference.batch_size=64
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

# ── Decision functions ──

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
    match = re.search(r'OUTPUT\s*:\s*\[\s*([\d\s,\-]+)\s*\]', input_str)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None


def resolve_chain(vec, df_choices):
    """Given input vec and decision function names, resolve to letter tuple."""
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
    """Apply N functions in sequence. Returns list of (trace_str, result_vec)."""
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

    seen = {}
    unique_inputs = []
    for ex in raw_data:
        input_str = ex["input"]
        if input_str not in seen:
            vec = extract_input_vector(input_str)
            chain_len = len(ex["letters"])
            seen[input_str] = len(unique_inputs)
            unique_inputs.append((input_str, chain_len, vec))

    log.info(f"Loaded {len(raw_data)} examples, {len(unique_inputs)} unique inputs")

    len_counts = Counter(cl for _, cl, _ in unique_inputs)
    for cl in sorted(len_counts):
        log.info(f"  Length {cl}: {len_counts[cl]} unique inputs")

    return unique_inputs, metadata, raw_data


# ── Ground truth computation ──

def compute_all_gt_traces(vec, chain_length):
    """For a single input vector, compute all 2^N GT traces."""
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
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def get_top_k_alternatives(logits, tokenizer, k=5):
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))
    return results


@torch.no_grad()
def generate_with_scores(model, tokenizer, prompts, cfg):
    """Batched free generation with per-step logit extraction."""
    batch_size = cfg.inference.batch_size
    search_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]
    semicolon_id = tokenizer.encode(";", add_special_tokens=False)[0]
    letter_to_tid, ordered_tids = build_letter_token_map(tokenizer)
    ordered_tids_tensor = torch.tensor(ordered_tids, dtype=torch.long, device="cuda")
    pad_id = tokenizer.pad_token_id

    # STOP token
    stop_tids = tokenizer.encode("STOP", add_special_tokens=False)
    stop_token_id = stop_tids[0] if stop_tids else None

    predictions = []
    per_step_probs = []
    per_step_gen_letters = []
    token_level_data = []
    all_sequences = []
    all_gen_ids = []  # for STOP detection

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
        scores = outputs.scores

        for i, seq in enumerate(sequences):
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

            clean_seq = seq
            while clean_seq and clean_seq[0] == pad_id:
                clean_seq = clean_seq[1:]
            all_sequences.append(clean_seq)

            gen_part = seq[padded_prompt_len:]
            eos_id = tokenizer.eos_token_id
            if eos_id in gen_part:
                gen_part = gen_part[:gen_part.index(eos_id)]
            all_gen_ids.append(gen_part)

            decision_points = [0]
            for step_idx, tok in enumerate(gen_part):
                if tok == semicolon_id:
                    decision_points.append(step_idx + 1)

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

            example_token_data = []
            for tok_idx, tok_id in enumerate(gen_part):
                if tok_idx < len(scores):
                    logits = scores[tok_idx][i]
                    entropy = compute_token_entropy(logits)
                    probs = torch.softmax(logits.float(), dim=-1)
                    token_prob = probs[tok_id].item()
                    top_k = get_top_k_alternatives(logits, tokenizer, k=5)
                    token_str = tokenizer.decode([tok_id])

                    # Add P(STOP) for this token position
                    p_stop = probs[stop_token_id].item() if stop_token_id is not None else 0.0

                    example_token_data.append({
                        "token": token_str,
                        "token_id": tok_id,
                        "entropy": entropy,
                        "probability": token_prob,
                        "top_k": top_k,
                        "is_decision_point": tok_idx in decision_points,
                        "p_stop": p_stop,
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
                        "p_stop": 0.0,
                    })

            token_level_data.append(example_token_data)

    return (predictions, per_step_probs, per_step_gen_letters,
            token_level_data, all_sequences, all_gen_ids)


# ── STOP probability probing ──

@torch.no_grad()
def probe_stop_probabilities(model, tokenizer, unique_inputs, raw_data, cfg):
    """
    Construct wrong-branch and correct-branch prefixes for test examples,
    run forward pass, extract P(STOP) at the next-token position.

    Returns:
        stop_metrics: dict of aggregated STOP metrics
        wrong_probe_details: list of dicts with per-probe info
        correct_probe_details: list of dicts with per-probe info
    """
    stop_tids = tokenizer.encode("STOP", add_special_tokens=False)
    stop_token_id = stop_tids[0] if stop_tids else None
    if stop_token_id is None:
        log.warning("STOP token not in vocabulary, skipping probing")
        return {}, [], []

    split_str = cfg.data.split_str
    batch_size = getattr(cfg.eval, 'stop_probe_batch_size', 64)

    # Build raw_test lookup by input string
    raw_by_input = {}
    for ex in raw_data:
        input_str = ex["input"]
        if input_str not in raw_by_input and not ex.get("is_negative", False):
            raw_by_input[input_str] = ex

    wrong_probes = []  # (prefix_ids, step_idx, example_idx, info_dict)
    correct_probes = []

    for i, (input_str, chain_len, input_vec) in enumerate(unique_inputs):
        if input_vec is None:
            continue

        raw_entry = raw_by_input.get(input_str)
        if raw_entry is None:
            continue

        letters = raw_entry.get("letters", [])
        df_names = raw_entry.get("decision_funcs", [])
        if not letters or not df_names:
            continue

        # Compute intermediate vectors
        current = list(input_vec)
        intermediate_vecs = [list(current)]
        for s in range(len(letters)):
            _, func = LETTER_TO_FUNC[letters[s]]
            current = func(list(current), K, [])
            intermediate_vecs.append(list(current))

        for s in range(len(letters)):
            vec_before = intermediate_vecs[s]
            idx_f = decision_func_f(vec_before)
            idx_g = decision_func_g(vec_before)

            # CORRECT PROBE: after step s, measure P(STOP)
            correct_steps = apply_chain_n(input_vec, letters[:s + 1])
            correct_trace = " ; ".join(step[0] for step in correct_steps) + " ;"
            prefix_text = f"{tokenizer.bos_token} {input_str} {split_str} {correct_trace}"
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            correct_probes.append((prefix_ids, s, i, {
                "input": input_str, "step": s, "chain_len": chain_len,
            }))

            # WRONG PROBE: opposite decision function
            if idx_f == idx_g:
                continue

            gt_df = df_names[s]
            wrong_df = "g" if gt_df == "f" else "f"
            wrong_idx = DECISION_FUNCS[wrong_df](vec_before)
            wrong_letter = INT_TO_LETTER[wrong_idx]

            if wrong_letter == letters[s]:
                continue

            if s > 0:
                prefix_steps = apply_chain_n(input_vec, letters[:s])
                prefix_traces = [step[0] for step in prefix_steps]
            else:
                prefix_traces = []

            _, wrong_func = LETTER_TO_FUNC[wrong_letter]
            trace_list = []
            wrong_func(list(vec_before), K, trace_list)
            wrong_trace_str = trace_list[0].replace(
                LETTER_TO_FUNC[wrong_letter][0], wrong_letter, 1
            )

            all_parts = prefix_traces + [wrong_trace_str]
            wrong_trace = " ; ".join(all_parts) + " ;"
            prefix_text = f"{tokenizer.bos_token} {input_str} {split_str} {wrong_trace}"
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            wrong_probes.append((prefix_ids, s, i, {
                "input": input_str, "step": s, "chain_len": chain_len,
                "gt_letter": letters[s], "wrong_letter": wrong_letter,
            }))

    log.info(f"Built {len(wrong_probes)} wrong probes, {len(correct_probes)} correct probes")

    # Run probes
    def run_probes(probes):
        p_stop_values = []
        pad_id = tokenizer.pad_token_id

        for b in range(0, len(probes), batch_size):
            batch = probes[b:b + batch_size]
            prefix_ids_list = [p[0] for p in batch]

            max_len = max(len(ids) for ids in prefix_ids_list)
            padded = []
            attention_masks = []
            for ids in prefix_ids_list:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            input_ids = torch.tensor(padded, dtype=torch.long, device="cuda")
            attention_mask = torch.tensor(attention_masks, dtype=torch.long, device="cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            for j in range(len(batch)):
                last_pos = max_len - 1
                token_logits = logits[j, last_pos, :]
                probs = torch.softmax(token_logits.float(), dim=-1)
                p_stop = probs[stop_token_id].item()
                p_stop_values.append(p_stop)

        return p_stop_values

    stop_metrics = {}
    wrong_details = []
    correct_details = []

    if wrong_probes:
        log.info(f"Running {len(wrong_probes)} wrong-branch probes...")
        p_stop_wrong = run_probes(wrong_probes)
        stop_metrics["avg_P_stop_after_wrong"] = float(np.mean(p_stop_wrong))
        stop_metrics["median_P_stop_after_wrong"] = float(np.median(p_stop_wrong))

        per_step_wrong = defaultdict(list)
        per_len_wrong = defaultdict(list)
        for (_, step, ex_idx, info), p in zip(wrong_probes, p_stop_wrong):
            per_step_wrong[step].append(p)
            per_len_wrong[info["chain_len"]].append(p)
            wrong_details.append({**info, "P_stop": p})

        for s in sorted(per_step_wrong):
            stop_metrics[f"step{s}_P_stop_after_wrong"] = float(np.mean(per_step_wrong[s]))
        for cl in sorted(per_len_wrong):
            stop_metrics[f"P_stop_after_wrong_len{cl}"] = float(np.mean(per_len_wrong[cl]))

    if correct_probes:
        log.info(f"Running {len(correct_probes)} correct-branch probes...")
        p_stop_correct = run_probes(correct_probes)
        stop_metrics["avg_P_stop_after_correct"] = float(np.mean(p_stop_correct))
        stop_metrics["median_P_stop_after_correct"] = float(np.median(p_stop_correct))

        per_step_correct = defaultdict(list)
        per_len_correct = defaultdict(list)
        for (_, step, ex_idx, info), p in zip(correct_probes, p_stop_correct):
            per_step_correct[step].append(p)
            per_len_correct[info["chain_len"]].append(p)
            correct_details.append({**info, "P_stop": p})

        for s in sorted(per_step_correct):
            stop_metrics[f"step{s}_P_stop_after_correct"] = float(np.mean(per_step_correct[s]))
        for cl in sorted(per_len_correct):
            stop_metrics[f"P_stop_after_correct_len{cl}"] = float(np.mean(per_len_correct[cl]))

    if "avg_P_stop_after_wrong" in stop_metrics and "avg_P_stop_after_correct" in stop_metrics:
        stop_metrics["stop_discrimination"] = (
            stop_metrics["avg_P_stop_after_wrong"] - stop_metrics["avg_P_stop_after_correct"]
        )

    return stop_metrics, wrong_details, correct_details


# ── Embedding extraction ──
# Token IDs may differ from extended experiment due to STOP token insertion.
# Look up dynamically instead of using hardcoded IDs.

def find_token_id(tokenizer, token_str):
    """Get token ID for a string, handling potential [UNK]."""
    tids = tokenizer.encode(token_str, add_special_tokens=False)
    return tids[0] if tids else None


@torch.no_grad()
def extract_last_layer_embeddings(model, tokenizer, prompts, all_sequences, cfg):
    """Extract last-layer hidden states at OUTPUT and intermediate vector positions."""
    batch_size = cfg.inference.batch_size
    embed_idx = cfg.inference.embed_intermediate_idx
    n_embd = cfg.model.n_embd

    output_tid = find_token_id(tokenizer, "OUTPUT")
    bracket_open_tid = find_token_id(tokenizer, "[")
    bracket_close_tid = find_token_id(tokenizer, "]")
    eos_tid = tokenizer.eos_token_id

    N = len(all_sequences)

    def find_output_positions(token_ids):
        try:
            output_pos = token_ids.index(output_tid)
        except ValueError:
            return None
        open_pos = None
        for j in range(output_pos + 1, len(token_ids)):
            if token_ids[j] == bracket_open_tid:
                open_pos = j
                break
        if open_pos is None:
            return None
        close_pos = None
        for j in range(open_pos + 1, len(token_ids)):
            if token_ids[j] == bracket_close_tid:
                close_pos = j
                break
        if close_pos is None:
            return None
        return list(range(open_pos, close_pos + 1))

    def find_intermediate_positions(token_ids, prompt_len, embed_idx_val=0):
        gen_part = token_ids[prompt_len:]
        bracket_pairs = []
        i = 0
        while i < len(gen_part):
            if gen_part[i] == bracket_open_tid:
                for j in range(i + 1, len(gen_part)):
                    if gen_part[j] == eos_tid:
                        break
                    if gen_part[j] == bracket_close_tid:
                        bracket_pairs.append((prompt_len + i, prompt_len + j))
                        i = j
                        break
            i += 1
        if not bracket_pairs:
            return None
        idx_val = max(0, min(embed_idx_val, len(bracket_pairs) - 1))
        open_pos, close_pos = bracket_pairs[idx_val]
        return list(range(open_pos, close_pos + 1))

    first_output_pos = find_output_positions(all_sequences[0])
    vec_token_count = len(first_output_pos) if first_output_pos else 13
    total_tokens = vec_token_count * 2

    all_embeddings = np.full((N, total_tokens, n_embd), np.nan, dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for b in trange(0, N, batch_size, desc="Extracting embeddings"):
        batch_seqs = all_sequences[b:b + batch_size]
        batch_prompts = prompts[b:b + batch_size]
        bs = len(batch_seqs)

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

        last_hidden = outputs.hidden_states[-1]

        for i in range(bs):
            seq = batch_seqs[i]
            prompt_len = len(batch_prompts[i])
            pad_offset = pad_offsets[i]

            output_pos = find_output_positions(seq)
            intermediate_pos = find_intermediate_positions(seq, prompt_len, embed_idx)

            if output_pos is None or intermediate_pos is None:
                continue
            if len(output_pos) != vec_token_count or len(intermediate_pos) != vec_token_count:
                continue

            for j, pos in enumerate(output_pos):
                all_embeddings[b + i, j, :] = (
                    last_hidden[i, pos + pad_offset, :].float().cpu().numpy()
                )
            for j, pos in enumerate(intermediate_pos):
                all_embeddings[b + i, vec_token_count + j, :] = (
                    last_hidden[i, pos + pad_offset, :].float().cpu().numpy()
                )
            valid_mask[b + i] = True

        del outputs, last_hidden
        torch.cuda.empty_cache()

    return all_embeddings, valid_mask


# ── Metrics ──

def compute_metrics(unique_inputs, predictions, per_step_probs, per_step_gen_letters,
                    metadata, token_level_data=None, all_gen_ids=None, stop_token_id=None):
    """Compute all metrics including STOP detection in free generation."""

    train_tuples_per_len = {}
    test_tuples_per_len = {}
    for length_str, tuples in metadata["train_tuples_per_length"].items():
        train_tuples_per_len[int(length_str)] = set(tuple(t) for t in tuples)
    for length_str, tuples in metadata["test_tuples_per_length"].items():
        test_tuples_per_len[int(length_str)] = set(tuple(t) for t in tuples)

    n = len(unique_inputs)

    n_way_correct = 0
    parseable_count = 0
    all_blocks_valid_count = 0
    valid_chain_count = 0
    valid_solution_count = 0
    all_steps_valid_count = 0

    # STOP detection counters
    stop_in_gen_count = 0

    n_way_per_len = defaultdict(lambda: {"correct": 0, "total": 0})
    parseable_per_len = defaultdict(lambda: {"count": 0, "total": 0})
    valid_chain_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    valid_solution_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    all_steps_valid_per_len = defaultdict(lambda: {"valid": 0, "total": 0})
    per_step_valid = defaultdict(int)
    per_step_total = defaultdict(int)

    chosen_train = 0
    chosen_test = 0
    chosen_neither = 0
    chosen_train_per_len = defaultdict(int)
    chosen_test_per_len = defaultdict(int)

    prob_f_per_step = defaultdict(list)
    prob_g_per_step = defaultdict(list)
    decision_mass_per_step = defaultdict(list)

    per_func_per_step = defaultdict(lambda: {"correct": 0, "total": 0})
    tuple_frequency = Counter()
    tuple_frequency_per_len = defaultdict(Counter)

    per_example_results = []

    vis_per_step_probs = []
    vis_per_step_idx_f = []
    vis_per_step_idx_g = []
    vis_gen_letters = []
    vis_vecs = []
    vis_is_valid_chain = []
    vis_is_valid_solution = []
    vis_has_different_final = []

    for i, (input_str, expected_chain_len, input_vec) in enumerate(unique_inputs):
        pred = predictions[i].strip()
        step_probs = per_step_probs[i]
        gen_letters = per_step_gen_letters[i]

        # STOP detection
        if all_gen_ids is not None and stop_token_id is not None:
            if stop_token_id in all_gen_ids[i]:
                stop_in_gen_count += 1

        n_way_per_len[expected_chain_len]["total"] += 1
        parseable_per_len[expected_chain_len]["total"] += 1

        gt_traces = compute_all_gt_traces(input_vec, expected_chain_len)

        pred_blocks_for_nway = parse_chain_output_n(pred)
        pred_letters = tuple(b["letter"] for b in pred_blocks_for_nway if b is not None) if pred_blocks_for_nway else ()

        matched_gt = None
        for gt in gt_traces:
            if pred_letters == tuple(gt["letters"]):
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

        pred_blocks = parse_chain_output_n(pred)
        is_parseable = pred_blocks is not None and len(pred_blocks) >= 1
        if is_parseable:
            parseable_count += 1
            parseable_per_len[expected_chain_len]["count"] += 1

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

            if expected_result is not None and block_valid:
                current_vec = expected_result
            elif block["vec"] is not None:
                current_vec = block["vec"]
            else:
                current_vec = None

        if steps_valid and all(steps_valid):
            all_blocks_valid_count += 1

        valid_chain_per_len[expected_chain_len]["total"] += 1
        valid_solution_per_len[expected_chain_len]["total"] += 1
        steps_chose_fg = []
        check_vec = list(input_vec)
        final_computed_vec = None

        for s in range(n_pred_steps):
            if pred_blocks and s < len(pred_blocks) and pred_blocks[s] is not None:
                chosen_letter = pred_blocks[s]["letter"]
                if check_vec is not None and len(check_vec) >= 5:
                    idx_f = decision_func_f(check_vec)
                    idx_g = decision_func_g(check_vec)
                    letter_f = INT_TO_LETTER[idx_f]
                    letter_g = INT_TO_LETTER[idx_g]

                    chose_fg = (chosen_letter == letter_f or chosen_letter == letter_g)
                    steps_chose_fg.append(chose_fg)

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

        all_chose_fg = len(steps_chose_fg) == n_pred_steps and all(steps_chose_fg) if steps_chose_fg else False
        all_valid = steps_valid and all(steps_valid) if steps_valid else False
        is_all_steps_valid = all_chose_fg and all_valid and n_pred_steps > 0

        correct_length = n_pred_steps == expected_chain_len
        is_valid_chain = correct_length and is_all_steps_valid
        is_valid_solution = False
        has_different_final = False

        if is_all_steps_valid:
            all_steps_valid_count += 1
            all_steps_valid_per_len[expected_chain_len]["valid"] += 1
        all_steps_valid_per_len[expected_chain_len]["total"] += 1

        if is_valid_chain:
            valid_chain_count += 1
            valid_chain_per_len[expected_chain_len]["valid"] += 1

            output_vec = extract_output_vector(input_str)
            if output_vec is not None and final_computed_vec is not None:
                if list(final_computed_vec) == list(output_vec):
                    is_valid_solution = True
                    valid_solution_count += 1
                    valid_solution_per_len[expected_chain_len]["valid"] += 1
                else:
                    has_different_final = True

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

        vis_vecs.append(list(input_vec))
        vis_per_step_probs.append([p.tolist() if hasattr(p, 'tolist') else list(p) for p in step_probs])
        vis_per_step_idx_f.append(step_idx_f_list)
        vis_per_step_idx_g.append(step_idx_g_list)
        vis_gen_letters.append(list(gen_letters))
        vis_is_valid_chain.append(is_valid_chain)
        vis_is_valid_solution.append(is_valid_solution)
        vis_has_different_final.append(has_different_final)

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
            for s in range(len(step_probs)):
                if s < len(step_idx_f_list) and step_idx_f_list[s] is not None:
                    probs = step_probs[s]
                    if not np.isnan(probs).any():
                        row[f"step{s+1}_P_f"] = float(probs[step_idx_f_list[s]])
                        row[f"step{s+1}_P_g"] = float(probs[step_idx_g_list[s]])
            per_example_results.append(row)

    # Aggregate metrics
    metrics = {
        "n_way_match": n_way_correct / max(n, 1),
        "all_blocks_valid": all_blocks_valid_count / max(parseable_count, 1),
        "parseable_fraction": parseable_count / max(n, 1),

        "chosen_train_tuple_frac": chosen_train / max(parseable_count, 1),
        "chosen_test_tuple_frac": chosen_test / max(parseable_count, 1),
        "chosen_neither_frac": chosen_neither / max(parseable_count, 1),

        "valid_chain_frac": valid_chain_count / max(n, 1),
        "num_valid_chains": valid_chain_count,
        "valid_solution_frac": valid_solution_count / max(n, 1),
        "num_valid_solutions": valid_solution_count,
        "all_steps_valid_frac": all_steps_valid_count / max(n, 1),
        "num_all_steps_valid": all_steps_valid_count,

        "num_unique_inputs": n,
        "num_parseable": parseable_count,
        "num_n_way_correct": n_way_correct,

        # STOP detection
        "stop_in_free_gen_rate": stop_in_gen_count / max(n, 1),
        "stop_in_free_gen_count": stop_in_gen_count,
    }

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

        vc = valid_chain_per_len[cl]
        if vc["total"] > 0:
            metrics[f"valid_chain_frac_len{cl}"] = vc["valid"] / vc["total"]
        vs = valid_solution_per_len[cl]
        if vs["total"] > 0:
            metrics[f"valid_solution_frac_len{cl}"] = vs["valid"] / vs["total"]

        asv = all_steps_valid_per_len[cl]
        if asv["total"] > 0:
            metrics[f"all_steps_valid_frac_len{cl}"] = asv["valid"] / asv["total"]

    for s in range(5):
        if per_step_total[s] > 0:
            metrics[f"step{s+1}_block_valid"] = per_step_valid[s] / per_step_total[s]

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

    tuple_dist = {}
    for cl in sorted(tuple_frequency_per_len):
        top = tuple_frequency_per_len[cl].most_common(20)
        tuple_dist[str(cl)] = {",".join(t): c for t, c in top}
    metrics["tuple_distribution_per_length"] = tuple_dist

    vis_data = {
        "per_step_probs": vis_per_step_probs,
        "vecs": vis_vecs,
        "per_step_idx_f": vis_per_step_idx_f,
        "per_step_idx_g": vis_per_step_idx_g,
        "gen_letters": vis_gen_letters,
        "token_level_data": token_level_data,
        "is_valid_chain": vis_is_valid_chain,
        "is_valid_solution": vis_is_valid_solution,
        "has_different_final": vis_has_different_final,
    }

    return metrics, per_example_results, vis_data


# ── Main ──

@hydra.main(config_path="../config", config_name="base_decision_chains_stop", version_base=None)
def main(cfg: DictConfig):
    log.info("STOP token experiment inference: free generation + STOP analysis")

    # 1. Load
    model, tokenizer = load_model_and_tokenizer(cfg)
    unique_inputs, metadata, raw_data = load_test_data(cfg)

    # Verify STOP token
    stop_tids = tokenizer.encode("STOP", add_special_tokens=False)
    stop_token_id = stop_tids[0] if stop_tids else None
    if stop_token_id is not None:
        log.info(f"STOP token ID: {stop_token_id}")
    else:
        log.warning("STOP token not found in vocabulary!")

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
    (predictions, step_probs, step_gen_letters,
     token_level_data, all_sequences, all_gen_ids) = generate_with_scores(
        model, tokenizer, prompts, cfg
    )

    # 3b. Extract embeddings
    log.info("Extracting last-layer embeddings...")
    embeddings, embed_valid_mask = extract_last_layer_embeddings(
        model, tokenizer, prompts, all_sequences, cfg
    )
    log.info(f"Embeddings extracted: shape {embeddings.shape}, "
             f"valid: {embed_valid_mask.sum()}/{len(embed_valid_mask)}")

    # 3c. STOP probability probing (before freeing model)
    log.info("Running STOP probability probing...")
    stop_metrics, wrong_details, correct_details = probe_stop_probabilities(
        model, tokenizer, unique_inputs, raw_data, cfg
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # 4. Compute standard metrics + STOP detection
    metrics, per_example_results, vis_data = compute_metrics(
        unique_inputs, predictions, step_probs, step_gen_letters, metadata,
        token_level_data=token_level_data,
        all_gen_ids=all_gen_ids,
        stop_token_id=stop_token_id,
    )

    # Merge STOP probing metrics
    metrics.update(stop_metrics)

    # 5. Print summary
    print("\n" + "=" * 60)
    print("  STOP Token Experiment Inference Results")
    print("=" * 60)

    print(f"\nOverall ({metrics['num_unique_inputs']} unique inputs):")
    print(f"  {'n_way_match':35s}: {metrics['n_way_match']:.4f}")
    print(f"  {'all_blocks_valid':35s}: {metrics['all_blocks_valid']:.4f}")
    print(f"  {'parseable_fraction':35s}: {metrics['parseable_fraction']:.4f}")

    print(f"\nSTOP detection in free generation:")
    print(f"  {'stop_in_free_gen_rate':35s}: {metrics['stop_in_free_gen_rate']:.4f}")
    print(f"  {'stop_in_free_gen_count':35s}: {metrics['stop_in_free_gen_count']}")

    if "avg_P_stop_after_wrong" in metrics:
        print(f"\nSTOP probability probing:")
        print(f"  {'avg_P_stop_after_wrong':35s}: {metrics['avg_P_stop_after_wrong']:.4f}")
        print(f"  {'avg_P_stop_after_correct':35s}: {metrics.get('avg_P_stop_after_correct', 0):.4f}")
        print(f"  {'stop_discrimination':35s}: {metrics.get('stop_discrimination', 0):.4f}")

        print(f"\n  Per-step STOP probing:")
        for s in range(5):
            w_key = f"step{s}_P_stop_after_wrong"
            c_key = f"step{s}_P_stop_after_correct"
            if w_key in metrics:
                print(f"    Step {s}: P_wrong={metrics[w_key]:.4f}  "
                      f"P_correct={metrics.get(c_key, 0):.4f}")

    print(f"\nPer chain length:")
    for cl in [3, 4, 5]:
        key = f"n_way_match_len{cl}"
        if key in metrics:
            n_key = f"n_len{cl}"
            print(f"  Length {cl} (n={metrics[n_key]}):")
            print(f"    {'n_way_match':31s}: {metrics[key]:.4f}")
            pf = metrics.get(f"parseable_frac_len{cl}", 0)
            print(f"    {'parseable_fraction':31s}: {pf:.4f}")

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

    print(f"\nValid chain / solution metrics:")
    print(f"  {'valid_chain_frac':35s}: {metrics['valid_chain_frac']:.4f}")
    print(f"  {'valid_solution_frac':35s}: {metrics['valid_solution_frac']:.4f}")
    print(f"  {'all_steps_valid_frac':35s}: {metrics['all_steps_valid_frac']:.4f}")

    # 6. Save
    output_dir = Path(to_absolute_path(cfg.eval.results_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": {k: v for k, v in metrics.items()
                    if k != "tuple_distribution_per_length"},
        "tuple_distribution_per_length": metrics.get("tuple_distribution_per_length", {}),
        "per_example_results": per_example_results,
        "stop_probe_wrong_details": wrong_details[:100],
        "stop_probe_correct_details": correct_details[:100],
    }
    results_path = output_dir / "inference_results_stop.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save probability arrays
    np.savez(
        output_dir / "decision_point_probs_stop.npz",
        per_step_probs=np.array(vis_data["per_step_probs"], dtype=object),
        unique_vecs=np.array(vis_data["vecs"]),
        per_step_idx_f=np.array(vis_data["per_step_idx_f"], dtype=object),
        per_step_idx_g=np.array(vis_data["per_step_idx_g"], dtype=object),
        gen_letters_all=np.array(vis_data["gen_letters"], dtype=object),
        # STOP-specific
        stop_detected=np.array([stop_token_id in gids for gids in all_gen_ids]
                               if stop_token_id else [False] * len(all_gen_ids), dtype=bool),
    )
    log.info(f"Probability arrays saved to {output_dir / 'decision_point_probs_stop.npz'}")

    # Label trajectories
    labeled_path = output_dir / "labeled_dataset.jsonl"
    label_trajectories(unique_inputs, predictions, str(labeled_path))

    # Save labeled embeddings
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
    log.info(f"Labeled embeddings saved to {npz_path}")

    # HTML visualization
    from visualization.visualize_superposition import generate_html_extended

    html_path = output_dir / "letter_probs_stop.html"
    generate_html_extended(
        vis_data["per_step_probs"],
        vis_data["vecs"],
        str(html_path),
        per_step_idx_f=vis_data["per_step_idx_f"],
        per_step_idx_g=vis_data["per_step_idx_g"],
        gen_letters_all=vis_data["gen_letters"],
        train_tuples_per_length=metadata["train_tuples_per_length"],
        token_level_data=vis_data.get("token_level_data"),
        unique_inputs=unique_inputs,
        predictions=predictions,
        metrics=metrics,
        is_valid_chain=vis_data.get("is_valid_chain"),
        is_valid_solution=vis_data.get("is_valid_solution"),
        has_different_final=vis_data.get("has_different_final"),
    )


if __name__ == "__main__":
    main()
