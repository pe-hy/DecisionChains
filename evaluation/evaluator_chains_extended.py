"""
Extended chain evaluator for 3-5 function chains.

Runs free generation with output_scores=True to capture probability
distributions at each decision point (letter position). At each step,
measures P(letter_f) and P(letter_g) — the probabilities assigned to the
letters that decision functions f and g would select given the current
intermediate vector.

Also computes block validity metrics (same as ChainEvaluator).
"""

import sys
import os
import re
import json
import torch
import numpy as np
import wandb
from tqdm import trange
from collections import defaultdict
from transformers import PreTrainedTokenizerFast

from ops.transformations import (
    reverse, add_1, double, negate, cumsum, scale_by_first,
    rotate_left, swap_pairs, position_multiply, diff,
    add_last_to_all, square, rotate_right, add_first_to_all, cumsum_reverse,
    prefix_product, sliding_sum, position_add, conditional_double,
    interleave_sum_diff,
)

K = 10

LETTER_TO_FUNC = {
    "a": ("reverse", reverse),
    "b": ("add_1", add_1),
    "c": ("double", double),
    "d": ("negate", negate),
    "e": ("cumsum", cumsum),
    "f": ("scale_by_first", scale_by_first),
    "g": ("rotate_left", rotate_left),
    "h": ("swap_pairs", swap_pairs),
    "i": ("position_multiply", position_multiply),
    "j": ("diff", diff),
    "k": ("add_last_to_all", add_last_to_all),
    "l": ("square", square),
    "m": ("rotate_right", rotate_right),
    "n": ("add_first_to_all", add_first_to_all),
    "o": ("cumsum_reverse", cumsum_reverse),
    "p": ("prefix_product", prefix_product),
    "q": ("sliding_sum", sliding_sum),
    "r": ("position_add", position_add),
    "s": ("conditional_double", conditional_double),
    "t": ("interleave_sum_diff", interleave_sum_diff),
}

LETTERS = list(LETTER_TO_FUNC.keys())  # a-t


# ── Decision functions ──

def is_even(x: int) -> int:
    return 1 if x % 2 == 0 else 0


def decision_func_f(L) -> int:
    return is_even(L[0]) * 10 + L[1]


def decision_func_g(L) -> int:
    return is_even(L[3]) * 10 + L[4]


INT_TO_LETTER = {i: chr(ord("a") + i) for i in range(20)}


# ── Parsing helpers ──

def extract_input_vector(input_str):
    """Extract the INPUT vector from 'INPUT : [ ... ] OUTPUT : [ ... ]'."""
    match = re.search(r'INPUT\s*:\s*\[\s*([\d\s,]+)\s*\]', input_str)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None


def extract_output_vector(input_str):
    """Extract the OUTPUT vector from 'INPUT : [ ... ] OUTPUT : [ ... ]'."""
    match = re.search(r'OUTPUT\s*:\s*\[\s*([\d\s,]+)\s*\]', input_str)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        return None


def parse_vector_from_text(text):
    """Extract the last [ d , d , ... ] vector from a text block."""
    matches = list(re.finditer(r'\[\s*([\d\s,]+)\s*\]', text))
    if not matches:
        return None
    try:
        return [int(x.strip()) for x in matches[-1].group(1).split(",")]
    except ValueError:
        return None


def parse_chain_output_n(output_str):
    """Parse chain output into N blocks split by ' ; '.

    Returns list of dicts with keys: letter, block, vec
    or None if parsing fails entirely.
    """
    parts = output_str.split(" ; ")
    if len(parts) < 1:
        return None

    blocks = []
    for part in parts:
        part = part.strip()
        if len(part) < 2 or part[1] != " ":
            blocks.append(None)
            continue

        letter = part[0]
        vec = parse_vector_from_text(part)
        blocks.append({
            "letter": letter,
            "block": part,
            "vec": vec,
        })

    return blocks


def apply_and_trace(letter, vec):
    """Apply function for `letter` to `vec`, return (result_vec, trace_block_str)."""
    entry = LETTER_TO_FUNC.get(letter)
    if entry is None:
        return None, None
    name, func = entry
    trace = []
    result = func(list(vec), K, trace)
    trace_str = trace[0].replace(name, letter, 1)
    return result, trace_str


class ExtendedChainEvaluator:
    """Evaluates extended chain predictions (3-5 steps) with probability tracking.

    Runs free generation with output_scores=True to capture logit distributions
    at each decision point. Measures P(letter_f) and P(letter_g) at each step.
    Also computes block validity metrics.
    """

    def __init__(self, config, tokenizer, split_str, hf_model, batch_size=512,
                 num_examples=512, raw_test_data=None):
        self.config = config
        self.tokenizer = tokenizer
        self.split_str = split_str
        self.hf_model = hf_model
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.raw_test_data = raw_test_data

        # Pre-compute letter token IDs in order a-t
        self.letter_token_ids = []
        for letter in LETTERS:
            tid = tokenizer.encode(letter, add_special_tokens=False)
            self.letter_token_ids.append(tid[0] if tid else None)
        self.letter_tid_tensor = torch.tensor(
            [t for t in self.letter_token_ids if t is not None]
        )

        self.semicolon_id = tokenizer.encode(";", add_special_tokens=False)[0]
        self.search_token_id = tokenizer.encode(split_str, add_special_tokens=False)[0]

    def get_prompts_and_gts(self, test_dataset):
        """Extract prompts (up to [TRACE]) and ground truths."""
        eos_id = self.tokenizer.eos_token_id

        prompts = []
        gts = []
        prompt_texts = []
        kept_indices = []

        for idx, sample in enumerate(test_dataset):
            ids = sample["input_ids"]
            try:
                split_idx = ids.index(self.search_token_id)
                end_idx = ids.index(eos_id) if eos_id in ids else len(ids)
            except ValueError:
                continue

            prompt_ids = ids[:split_idx + 1]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = self.tokenizer.bos_token + " " + prompt_text
            prompt_with_bos = self.tokenizer.encode(full_prompt, add_special_tokens=False)

            gt_ids = ids[split_idx + 1:end_idx]
            gt = self.tokenizer.decode(gt_ids, skip_special_tokens=True)

            prompts.append(prompt_with_bos)
            gts.append(gt)
            prompt_texts.append(prompt_text)
            kept_indices.append(idx)

        return prompts, gts, prompt_texts, kept_indices

    def generate_with_scores(self, prompts):
        """Run greedy generation with output_scores=True.

        Returns (predictions, all_scores, all_gen_ids) where:
        - predictions: list of decoded strings (after [TRACE])
        - all_scores: list of lists of score tensors per example
        - all_gen_ids: list of generated token ID lists per example
        """
        predictions = []
        all_scores = []
        all_gen_ids = []

        self.hf_model.cuda()
        self.hf_model.eval()

        for b in trange(0, len(prompts), self.batch_size,
                        desc="ExtChain eval (free+scores)"):
            batch = prompts[b:b + self.batch_size]
            batch_text = [self.tokenizer.decode(x, skip_special_tokens=False)
                          for x in batch]
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                batch_text, return_tensors="pt", padding=True
            ).to("cuda")

            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.hf_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=self.config.model.block_size,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # outputs.scores: tuple of (batch_size, vocab_size) per step
            scores = outputs.scores  # tuple of length n_generated_steps
            gen_sequences = outputs.sequences  # (batch_size, prompt_len + n_gen)

            batch_size_actual = gen_sequences.shape[0]

            for i in range(batch_size_actual):
                # Extract generated part (after prompt)
                full_ids = gen_sequences[i].tolist()
                gen_part = full_ids[prompt_len:]

                # Remove EOS/PAD from end
                eos_id = self.tokenizer.eos_token_id
                if eos_id in gen_part:
                    gen_part = gen_part[:gen_part.index(eos_id)]

                # Decode prediction
                try:
                    # Find [TRACE] in full sequence
                    split_index = full_ids.index(self.search_token_id)
                    end_index = prompt_len + len(gen_part)
                    after_delim = self.tokenizer.decode(
                        full_ids[split_index + 1:end_index],
                        skip_special_tokens=True
                    )
                except ValueError:
                    after_delim = self.tokenizer.decode(gen_part,
                                                        skip_special_tokens=True)

                predictions.append(after_delim)
                all_gen_ids.append(gen_part)

                # Extract per-example scores for generated steps
                example_scores = []
                for step_idx in range(len(gen_part)):
                    if step_idx < len(scores):
                        example_scores.append(scores[step_idx][i].cpu().float())
                all_scores.append(example_scores)

        return predictions, all_scores, all_gen_ids

    def compute_metrics(self, predictions, gts, prompt_texts,
                        all_scores, all_gen_ids, kept_indices=None):
        """Compute block validity + probability + valid chain/solution metrics."""
        n = len(predictions)
        letter_tid_tensor = self.letter_tid_tensor

        # Block validity counters
        exact_match = 0
        n_parseable = 0
        per_step_valid = defaultdict(int)       # step_idx -> count valid
        per_step_total = defaultdict(int)        # step_idx -> count total
        per_func_per_step = defaultdict(lambda: {"valid": 0, "total": 0})

        # Valid chain / valid solution counters
        valid_chain_count = 0
        valid_solution_count = 0

        # Per-step selection and op accuracy
        sel_correct_total = 0
        sel_total = 0
        op_correct_total = 0
        op_total = 0

        # Per-chain-length buckets
        len_sel = defaultdict(lambda: {"correct": 0, "total": 0})
        len_op = defaultdict(lambda: {"correct": 0, "total": 0})
        len_valid_chain = defaultdict(lambda: {"count": 0, "total": 0})
        len_valid_solution = defaultdict(lambda: {"count": 0, "total": 0})

        # Probability tracking
        # prob_f[step_idx] and prob_g[step_idx]: lists of floats
        prob_f_per_step = defaultdict(list)
        prob_g_per_step = defaultdict(list)

        example_rows = []

        for i in range(n):
            pred = predictions[i].strip()
            gt = gts[i].strip()
            gen_ids = all_gen_ids[i]
            scores = all_scores[i]

            if pred == gt:
                exact_match += 1

            # Parse input vector from prompt
            prompt_text = prompt_texts[i] if i < len(prompt_texts) else ""
            input_vec = extract_input_vector(prompt_text)

            # Parse predicted chain into blocks
            pred_blocks = parse_chain_output_n(pred)

            if pred_blocks is None or input_vec is None:
                if len(example_rows) < 50:
                    example_rows.append({
                        "input": prompt_text,
                        "gt": gt, "pred": pred, "exact": pred == gt,
                    })
                continue

            n_parseable += 1
            n_steps = len(pred_blocks)

            # Compute GT intermediates for selection/op tracking
            gt_letters_i = []
            gt_intermediates_i = None
            chain_len_i = None
            if self.raw_test_data is not None:
                raw_idx_gi = kept_indices[i] if kept_indices and i < len(kept_indices) else i
                if raw_idx_gi < len(self.raw_test_data):
                    re_i = self.raw_test_data[raw_idx_gi]
                    gt_letters_i = re_i.get("letters", [])
                    chain_len_i = len(gt_letters_i)
                    gt_intermediates_i = [list(input_vec)]
                    for gl in gt_letters_i:
                        _, gf = LETTER_TO_FUNC[gl]
                        gt_intermediates_i.append(gf(list(gt_intermediates_i[-1]), K, []))

            # Find decision points in generated token sequence:
            # step 0 = token index 0 (first generated token = letter1)
            # step s = token index after the s-th semicolon
            decision_points = [0]
            for step_idx, tok_id in enumerate(gen_ids):
                if tok_id == self.semicolon_id:
                    decision_points.append(step_idx + 1)

            # Validate blocks and extract probabilities
            current_vec = input_vec
            steps_valid = []
            steps_chose_fg = []
            final_computed_vec = None

            for s in range(n_steps):
                block = pred_blocks[s]
                per_step_total[s] += 1

                if block is None or current_vec is None:
                    steps_valid.append(False)
                    steps_chose_fg.append(False)
                    current_vec = None
                    continue

                letter = block["letter"]

                # Block validity: recompute trace and compare
                expected_result, expected_block = apply_and_trace(letter, current_vec)
                block_valid = (expected_block is not None
                               and block["block"] == expected_block)

                steps_valid.append(block_valid)
                if block_valid:
                    per_step_valid[s] += 1
                per_func_per_step[(s, letter)]["total"] += 1
                if block_valid:
                    per_func_per_step[(s, letter)]["valid"] += 1

                # Check if letter matches f or g decision function
                valid_fg = False
                if current_vec is not None and len(current_vec) >= 5:
                    idx_f = decision_func_f(current_vec)
                    idx_g = decision_func_g(current_vec)
                    if idx_f in INT_TO_LETTER and idx_g in INT_TO_LETTER:
                        letter_f = INT_TO_LETTER[idx_f]
                        letter_g = INT_TO_LETTER[idx_g]
                        valid_fg = True
                        steps_chose_fg.append(letter == letter_f or letter == letter_g)
                    else:
                        steps_chose_fg.append(False)
                else:
                    steps_chose_fg.append(False)

                # Probability extraction at this decision point
                if valid_fg and s < len(decision_points):
                    dp = decision_points[s]
                    if dp < len(scores):
                        logits = scores[dp]
                        probs = torch.softmax(logits, dim=-1)
                        letter_probs = probs[letter_tid_tensor].numpy()

                        prob_f_per_step[s].append(float(letter_probs[idx_f]))
                        prob_g_per_step[s].append(float(letter_probs[idx_g]))

                # Selection and op accuracy using GT intermediates
                if (gt_intermediates_i is not None and s < len(gt_letters_i)
                        and s < len(gt_intermediates_i)):
                    gt_letter = gt_letters_i[s]
                    sel_total += 1
                    if chain_len_i is not None:
                        len_sel[chain_len_i]["total"] += 1
                    if letter == gt_letter:
                        sel_correct_total += 1
                        if chain_len_i is not None:
                            len_sel[chain_len_i]["correct"] += 1
                    # Op: is arithmetic correct for model's letter against GT
                    # intermediate? (separates selection from execution)
                    exp_r, exp_b = apply_and_trace(letter, gt_intermediates_i[s])
                    op_total += 1
                    if chain_len_i is not None:
                        len_op[chain_len_i]["total"] += 1
                    if exp_b is not None and block["block"] == exp_b:
                        op_correct_total += 1
                        if chain_len_i is not None:
                            len_op[chain_len_i]["correct"] += 1

                # Advance current_vec for next step
                if expected_result is not None and block_valid:
                    current_vec = expected_result
                    final_computed_vec = current_vec
                elif block["vec"] is not None:
                    # Use parsed vector from model output even if block invalid
                    current_vec = block["vec"]
                    final_computed_vec = current_vec
                else:
                    current_vec = None

            # Valid chain: correct length + all f/g + all blocks valid
            expected_chain_len = chain_len_i  # already computed above

            all_chose_fg = steps_chose_fg and all(steps_chose_fg) and len(steps_chose_fg) == n_steps
            all_valid = steps_valid and all(steps_valid)
            correct_length = expected_chain_len is not None and n_steps == expected_chain_len
            is_valid_chain = correct_length and all_chose_fg and all_valid and n_steps > 0

            is_valid_solution = False
            if is_valid_chain:
                valid_chain_count += 1
                # Valid solution: valid chain + output matches
                output_vec = extract_output_vector(prompt_text)
                if output_vec is not None and final_computed_vec is not None:
                    if list(final_computed_vec) == list(output_vec):
                        valid_solution_count += 1
                        is_valid_solution = True

            # Per-chain-length tracking
            if chain_len_i is not None:
                len_valid_chain[chain_len_i]["total"] += 1
                len_valid_solution[chain_len_i]["total"] += 1
                if is_valid_chain:
                    len_valid_chain[chain_len_i]["count"] += 1
                if is_valid_solution:
                    len_valid_solution[chain_len_i]["count"] += 1

            if len(example_rows) < 50:
                example_rows.append({
                    "input": prompt_text,
                    "gt": gt, "pred": pred, "exact": pred == gt,
                    "n_steps": n_steps,
                })

        # Aggregate metrics
        metrics = {
            "exact_match": exact_match / max(n, 1),
            "parseable_fraction": n_parseable / max(n, 1),
            "valid_chain_frac": valid_chain_count / max(n, 1),
            "valid_solution_frac": valid_solution_count / max(n, 1),
        }

        # Per-step block validity
        for s in range(5):
            if per_step_total[s] > 0:
                metrics[f"step{s+1}_block_valid"] = (
                    per_step_valid[s] / per_step_total[s]
                )

        # Per-step probability metrics
        all_prob_f = []
        all_prob_g = []
        for s in range(5):
            if prob_f_per_step[s]:
                avg_f = float(np.nanmean(prob_f_per_step[s]))
                avg_g = float(np.nanmean(prob_g_per_step[s]))
                metrics[f"step{s+1}_avg_prob_f"] = avg_f
                metrics[f"step{s+1}_avg_prob_g"] = avg_g
                all_prob_f.extend(prob_f_per_step[s])
                all_prob_g.extend(prob_g_per_step[s])

        # Grand averages
        if all_prob_f:
            metrics["avg_prob_f"] = float(np.nanmean(all_prob_f))
        if all_prob_g:
            metrics["avg_prob_g"] = float(np.nanmean(all_prob_g))

        # Selection and op accuracy (aggregate)
        if sel_total > 0:
            metrics["selection_acc"] = sel_correct_total / sel_total
        if op_total > 0:
            metrics["op_acc"] = op_correct_total / op_total

        # Per-chain-length breakdown
        for L in sorted(len_sel.keys()):
            if len_sel[L]["total"] > 0:
                metrics[f"len{L}_selection_acc"] = len_sel[L]["correct"] / len_sel[L]["total"]
            if len_op[L]["total"] > 0:
                metrics[f"len{L}_op_acc"] = len_op[L]["correct"] / len_op[L]["total"]
            if len_valid_chain[L]["total"] > 0:
                metrics[f"len{L}_valid_chain_frac"] = (
                    len_valid_chain[L]["count"] / len_valid_chain[L]["total"]
                )
                metrics[f"len{L}_valid_solution_frac"] = (
                    len_valid_solution[L]["count"] / len_valid_solution[L]["total"]
                )

        return metrics, example_rows

    def evaluate(self, test_dataset):
        """Full evaluation pipeline."""
        raw = self.raw_test_data

        if len(test_dataset) > self.num_examples:
            indices = np.random.choice(len(test_dataset), self.num_examples,
                                       replace=False)
            test_dataset = test_dataset.select(indices.tolist())
            if raw is not None:
                raw = [raw[i] for i in indices]

        prompts, gts, prompt_texts, kept_indices = self.get_prompts_and_gts(
            test_dataset
        )

        if len(prompts) == 0:
            return {}, []

        predictions, all_scores, all_gen_ids = self.generate_with_scores(prompts)

        metrics, example_rows = self.compute_metrics(
            predictions, gts, prompt_texts, all_scores, all_gen_ids,
            kept_indices=kept_indices,
        )

        del self.hf_model
        torch.cuda.empty_cache()

        return metrics, example_rows
