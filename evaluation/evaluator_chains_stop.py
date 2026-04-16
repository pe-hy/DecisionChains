"""
STOP-aware chain evaluator for the STOP token experiment.

Extends ExtendedChainEvaluator with:
1. STOP detection in free generation
2. STOP probability probing: construct wrong-branch and correct-branch
   prefixes, measure P(STOP) at the next-token position.

Metrics added:
  - stop_in_free_gen_rate: fraction of free generations containing STOP
  - avg_P_stop_after_wrong: mean P(STOP) after wrong-branch step
  - avg_P_stop_after_correct: mean P(STOP) after correct step
  - stop_discrimination: P_wrong - P_correct
  - step{s}_P_stop_after_wrong / step{s}_P_stop_after_correct: per-step
"""

import torch
import numpy as np
from collections import defaultdict

from evaluation.evaluator_chains_extended import (
    ExtendedChainEvaluator,
    LETTER_TO_FUNC, LETTERS, INT_TO_LETTER, K,
    decision_func_f, decision_func_g,
    extract_input_vector,
)

DECISION_FUNCS = {
    "f": decision_func_f,
    "g": decision_func_g,
}


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


class StopChainEvaluator(ExtendedChainEvaluator):
    """Extends ExtendedChainEvaluator with STOP token detection and probing."""

    def __init__(self, config, tokenizer, split_str, hf_model, batch_size=512,
                 num_examples=512, raw_test_data=None):
        super().__init__(config, tokenizer, split_str, hf_model, batch_size,
                         num_examples, raw_test_data)

        # STOP token ID
        stop_tids = tokenizer.encode("STOP", add_special_tokens=False)
        self.stop_token_id = stop_tids[0] if stop_tids else None

        # Config for STOP probing
        self.stop_probe_examples = getattr(config.eval, 'stop_probe_examples', 256)
        self.stop_probe_batch_size = getattr(config.eval, 'stop_probe_batch_size', 64)

    def compute_metrics(self, predictions, gts, prompt_texts,
                        all_scores, all_gen_ids, kept_indices=None):
        """Extend parent metrics with STOP detection in free generation."""
        metrics, example_rows = super().compute_metrics(
            predictions, gts, prompt_texts, all_scores, all_gen_ids,
            kept_indices=kept_indices,
        )

        # STOP detection in free generation
        stop_in_gen_count = 0
        for gen_ids in all_gen_ids:
            if self.stop_token_id is not None and self.stop_token_id in gen_ids:
                stop_in_gen_count += 1

        n = len(predictions)
        metrics["stop_in_free_gen_rate"] = stop_in_gen_count / max(n, 1)
        metrics["stop_in_free_gen_count"] = float(stop_in_gen_count)

        return metrics, example_rows

    def _build_probes(self, raw_test_data, kept_indices):
        """
        Build wrong-branch and correct-branch probe prefixes.

        Wrong probes: [BOS] input [TRACE] correct_prefix ; wrong_step ;
          -> expect high P(STOP) at next token

        Correct probes: [BOS] input [TRACE] correct_prefix ;
          -> expect low P(STOP) at next token

        Returns:
            wrong_probes: list of (prefix_ids, step_idx, example_idx)
            correct_probes: list of (prefix_ids, step_idx, example_idx)
        """
        wrong_probes = []
        correct_probes = []

        n_probe = min(self.stop_probe_examples, len(raw_test_data))

        for i in range(n_probe):
            raw_idx = kept_indices[i] if kept_indices and i < len(kept_indices) else i
            if raw_idx >= len(raw_test_data):
                continue

            entry = raw_test_data[raw_idx]
            letters = entry.get("letters", [])
            df_names = entry.get("decision_funcs", [])
            input_str = entry.get("input", "")
            input_vec = extract_input_vector(input_str)

            if input_vec is None or not letters or not df_names:
                continue

            chain_len = len(letters)

            # Compute intermediate vectors
            current = list(input_vec)
            intermediate_vecs = [list(current)]
            for s in range(chain_len):
                _, func = LETTER_TO_FUNC[letters[s]]
                current = func(list(current), K, [])
                intermediate_vecs.append(list(current))

            for s in range(chain_len):
                vec_before = intermediate_vecs[s]
                idx_f = decision_func_f(vec_before)
                idx_g = decision_func_g(vec_before)

                # ── CORRECT PROBE: after step s, measure P(STOP) ──
                # Build prefix with correct steps 0..s
                correct_steps = apply_chain_n(input_vec, letters[:s + 1])
                correct_trace = " ; ".join(step[0] for step in correct_steps) + " ;"
                prefix_text = (f"{self.tokenizer.bos_token} {input_str} "
                               f"{self.split_str} {correct_trace}")
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                correct_probes.append((prefix_ids, s, i))

                # ── WRONG PROBE: at step s, use opposite decision function ──
                if idx_f == idx_g:
                    continue  # f and g agree, no valid wrong branch

                gt_df = df_names[s]
                wrong_df = "g" if gt_df == "f" else "f"
                wrong_idx = DECISION_FUNCS[wrong_df](vec_before)
                wrong_letter = INT_TO_LETTER[wrong_idx]

                if wrong_letter == letters[s]:
                    continue  # Should not happen, but safety check

                # Build wrong prefix: correct steps 0..s-1 + wrong step s + ";"
                if s > 0:
                    prefix_steps = apply_chain_n(input_vec, letters[:s])
                    prefix_traces = [step[0] for step in prefix_steps]
                else:
                    prefix_traces = []

                # Compute wrong step trace
                _, wrong_func = LETTER_TO_FUNC[wrong_letter]
                trace_list = []
                wrong_func(list(vec_before), K, trace_list)
                wrong_trace_str = trace_list[0].replace(
                    LETTER_TO_FUNC[wrong_letter][0], wrong_letter, 1
                )

                all_parts = prefix_traces + [wrong_trace_str]
                wrong_trace = " ; ".join(all_parts) + " ;"
                prefix_text = (f"{self.tokenizer.bos_token} {input_str} "
                               f"{self.split_str} {wrong_trace}")
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                wrong_probes.append((prefix_ids, s, i))

        return wrong_probes, correct_probes

    @torch.no_grad()
    def _run_probes(self, probes):
        """
        Run batched forward passes on probe prefixes.
        Extract P(STOP) at the last token position (next-token prediction).

        Returns list of P(STOP) values aligned with probes.
        """
        self.hf_model.eval()
        p_stop_values = []

        for b in range(0, len(probes), self.stop_probe_batch_size):
            batch = probes[b:b + self.stop_probe_batch_size]
            prefix_ids_list = [p[0] for p in batch]

            # Left-pad for batching
            max_len = max(len(ids) for ids in prefix_ids_list)
            pad_id = self.tokenizer.pad_token_id

            padded = []
            attention_masks = []
            for ids in prefix_ids_list:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            input_ids = torch.tensor(padded, dtype=torch.long, device="cuda")
            attention_mask = torch.tensor(attention_masks, dtype=torch.long, device="cuda")

            outputs = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # logits shape: (batch, seq_len, vocab_size)
            logits = outputs.logits

            for j in range(len(batch)):
                # Last real token position
                last_pos = max_len - 1  # since we left-padded, last position is always max_len-1
                token_logits = logits[j, last_pos, :]
                probs = torch.softmax(token_logits.float(), dim=-1)
                p_stop = probs[self.stop_token_id].item()
                p_stop_values.append(p_stop)

        return p_stop_values

    def probe_stop_probabilities(self, raw_test_data, kept_indices):
        """
        Construct wrong-branch and correct-branch prefixes,
        measure P(STOP), return aggregated metrics.
        """
        wrong_probes, correct_probes = self._build_probes(raw_test_data, kept_indices)

        metrics = {}

        if wrong_probes:
            p_stop_wrong = self._run_probes(wrong_probes)
            metrics["avg_P_stop_after_wrong"] = float(np.mean(p_stop_wrong))
            metrics["median_P_stop_after_wrong"] = float(np.median(p_stop_wrong))

            # Per-step breakdown
            per_step_wrong = defaultdict(list)
            for (_, step, _), p in zip(wrong_probes, p_stop_wrong):
                per_step_wrong[step].append(p)
            for s in sorted(per_step_wrong):
                metrics[f"step{s}_P_stop_after_wrong"] = float(np.mean(per_step_wrong[s]))

        if correct_probes:
            p_stop_correct = self._run_probes(correct_probes)
            metrics["avg_P_stop_after_correct"] = float(np.mean(p_stop_correct))
            metrics["median_P_stop_after_correct"] = float(np.median(p_stop_correct))

            per_step_correct = defaultdict(list)
            for (_, step, _), p in zip(correct_probes, p_stop_correct):
                per_step_correct[step].append(p)
            for s in sorted(per_step_correct):
                metrics[f"step{s}_P_stop_after_correct"] = float(np.mean(per_step_correct[s]))

        # Discrimination metric
        if "avg_P_stop_after_wrong" in metrics and "avg_P_stop_after_correct" in metrics:
            metrics["stop_discrimination"] = (
                metrics["avg_P_stop_after_wrong"] - metrics["avg_P_stop_after_correct"]
            )

        return metrics

    @torch.no_grad()
    def evaluate_stop_chains(self, raw_negatives, max_example_rows=25):
        """
        Measure stop-chain quality: for each negative example, force the
        wrong-branch prefix ("... ; wrong_step ;") and check that the
        model's free generation is exactly [STOP] (then EOS/end).

        The arithmetic of the wrong step is provided in the forced prefix
        (it's always correct in training data), so this isolates the
        question "does the model correctly emit STOP on a wrong branch?"

        Returns (metrics, example_rows). example_rows holds up to
        `max_example_rows` per-negative records for wandb display.
        """
        if not raw_negatives or self.stop_token_id is None:
            return {}, []

        self.hf_model.eval()
        self.hf_model.cuda()

        prefixes = []
        prefix_steps = []  # mistake_at for per-step breakdown
        prefix_lengths = []  # chain length for per-length breakdown
        prefix_meta = []   # (input_str, gt_output) parallel to prefixes
        for ex in raw_negatives:
            output = ex.get("output", "")
            if "STOP" not in output:
                continue
            # output = "correct_prefix ; wrong_step ; STOP"
            # strip trailing "STOP" → "correct_prefix ; wrong_step ;"
            prefix_out = output.rsplit("STOP", 1)[0].rstrip()
            input_str = ex.get("input", "")
            prefix_text = (f"{self.tokenizer.bos_token} {input_str} "
                           f"{self.split_str} {prefix_out}")
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefixes.append(prefix_ids)
            prefix_steps.append(ex.get("mistake_at", -1))
            prefix_lengths.append(ex.get("gt_chain_length", -1))
            prefix_meta.append((input_str, output))

        if not prefixes:
            return {}, []

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        correct = 0
        total = 0
        per_step_correct = defaultdict(int)
        per_step_total = defaultdict(int)
        len_stop = defaultdict(lambda: {"correct": 0, "total": 0})
        example_rows = []

        for b in range(0, len(prefixes), self.batch_size):
            batch = prefixes[b:b + self.batch_size]
            batch_steps = prefix_steps[b:b + self.batch_size]
            batch_lengths = prefix_lengths[b:b + self.batch_size]
            batch_meta = prefix_meta[b:b + self.batch_size]
            batch_text = [self.tokenizer.decode(p, skip_special_tokens=False)
                          for p in batch]
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                batch_text, return_tensors="pt", padding=True
            ).to("cuda")
            prompt_len = inputs["input_ids"].shape[1]

            outputs = self.hf_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=pad_id,
                max_new_tokens=4,
                num_beams=1,
                do_sample=False,
                eos_token_id=eos_id,
            )

            for i in range(outputs.shape[0]):
                gen = outputs[i, prompt_len:].tolist()
                while gen and gen[-1] in (eos_id, pad_id):
                    gen.pop()
                total += 1
                step = batch_steps[i]
                cl = batch_lengths[i]
                per_step_total[step] += 1
                if cl > 0:
                    len_stop[cl]["total"] += 1
                is_exact_stop = (gen == [self.stop_token_id])
                if is_exact_stop:
                    correct += 1
                    per_step_correct[step] += 1
                    if cl > 0:
                        len_stop[cl]["correct"] += 1

                if len(example_rows) < max_example_rows:
                    input_str, gt_output = batch_meta[i]
                    pred_text = self.tokenizer.decode(
                        gen, skip_special_tokens=True
                    )
                    example_rows.append({
                        "input": input_str,
                        "gt": gt_output,
                        "pred": pred_text,
                        "exact": is_exact_stop,
                        "n_steps": int(cl) if cl and cl > 0 else 0,
                        "variant": "unhealthy_stop",
                    })

        metrics = {
            "chain_quality_stop": correct / max(total, 1),
            "stop_chain_n_eval": float(total),
        }
        for s in sorted(per_step_total):
            if per_step_total[s] > 0 and s >= 0:
                metrics[f"step{s}_stop_chain_quality"] = (
                    per_step_correct[s] / per_step_total[s]
                )
        for cl in sorted(len_stop):
            if len_stop[cl]["total"] > 0:
                metrics[f"len{cl}_chain_quality_stop"] = (
                    len_stop[cl]["correct"] / len_stop[cl]["total"]
                )
        return metrics, example_rows

    def evaluate(self, test_dataset):
        """Split eval by is_negative: run GT chain quality on positives via
        free generation, run stop-chain quality on negatives via forced prefix,
        run STOP probing on positives, and emit a compact Health/ summary."""
        raw = self.raw_test_data

        if len(test_dataset) > self.num_examples:
            indices = np.random.choice(len(test_dataset), self.num_examples,
                                       replace=False)
            test_dataset = test_dataset.select(indices.tolist())
            if raw is not None:
                raw = [raw[i] for i in indices]

        # ── Split positives / negatives ──
        if raw is not None:
            pos_idx = [i for i, ex in enumerate(raw)
                       if not ex.get("is_negative", False)]
            neg_idx = [i for i, ex in enumerate(raw)
                       if ex.get("is_negative", False)]
            raw_pos = [raw[i] for i in pos_idx]
            raw_neg = [raw[i] for i in neg_idx]
            test_dataset_pos = (test_dataset.select(pos_idx)
                                if pos_idx else test_dataset.select([]))
        else:
            raw_pos, raw_neg = None, []
            test_dataset_pos = test_dataset

        # ── GT chain quality on positives (free generation) ──
        self.raw_test_data = raw_pos
        prompts, gts, prompt_texts, kept_indices = self.get_prompts_and_gts(
            test_dataset_pos
        )

        metrics = {}
        example_rows = []
        if len(prompts) > 0:
            predictions, all_scores, all_gen_ids = self.generate_with_scores(prompts)
            metrics, example_rows = self.compute_metrics(
                predictions, gts, prompt_texts, all_scores, all_gen_ids,
                kept_indices=kept_indices,
            )

        # Tag positive rows with variant so the merged wandb table has a
        # consistent schema with the STOP rows appended below.
        for r in example_rows:
            if "variant" not in r:
                r["variant"] = "healthy"

        # ── STOP chain quality on negatives (forced-prefix generation) ──
        stop_example_rows = []
        try:
            stop_chain_metrics, stop_example_rows = self.evaluate_stop_chains(
                raw_neg, max_example_rows=25
            )
            metrics.update(stop_chain_metrics)
        except Exception as e:
            print(f"WARNING: stop-chain eval failed: {e}")

        # Merge: keep up to 25 healthy rows + up to 25 STOP rows.
        example_rows = example_rows[:25] + stop_example_rows

        # Normalize schema: the extended evaluator's two row-emit branches
        # (early-return vs full-process) differ in whether they include
        # "n_steps", and STOP rows always include "variant". Union the keys
        # and fill missing fields with None so wandb.Table sees a flat schema.
        if example_rows:
            all_keys = set()
            for r in example_rows:
                all_keys.update(r.keys())
            for r in example_rows:
                for k in all_keys:
                    r.setdefault(k, None)

        # ── STOP probability probing on positives ──
        if raw_pos and self.stop_token_id is not None:
            try:
                stop_probe_metrics = self.probe_stop_probabilities(
                    raw_pos, list(range(len(raw_pos)))
                )
                metrics.update(stop_probe_metrics)
            except Exception as e:
                print(f"WARNING: STOP probing failed: {e}")

        # ── Health summary (headline metrics) ──
        cq_gt = metrics.get("valid_solution_frac", 0.0)
        cq_stop = metrics.get("chain_quality_stop", 0.0)
        no_spurious = 1.0 - metrics.get("stop_in_free_gen_rate", 0.0)
        stop_disc = metrics.get("stop_discrimination", 0.0)

        metrics["Health/chain_quality_gt"] = cq_gt
        metrics["Health/chain_quality_stop"] = cq_stop
        metrics["Health/no_spurious_stop"] = no_spurious
        metrics["Health/stop_discrimination"] = stop_disc
        metrics["Health/overall"] = max(0.0, min(1.0, cq_gt * cq_stop * no_spurious))

        del self.hf_model
        torch.cuda.empty_cache()

        return metrics, example_rows
