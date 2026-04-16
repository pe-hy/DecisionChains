"""
GRPO post-training for decision chain models.

Loads a pretrained HF checkpoint and applies Group Relative Policy
Optimization with binary correctness reward (correct chain = 1, else 0).

For each iteration:
  1. Sample a batch of unique inputs from the training data.
  2. Generate K completions per input using temperature sampling.
  3. Score each completion with check_completion_correct.
  4. Compute group-normalized advantages.
  5. Update the policy with a clipped policy gradient.

Key diagnostic: fraction of groups with mixed rewards (some correct, some wrong).
Groups where all K are correct or all wrong produce zero advantage = zero gradient.
This directly measures the "exploration ceiling".

Usage:
    python grpo_train.py
    python grpo_train.py grpo.group_size=32 grpo.temperature=1.0
    python grpo_train.py grpo.pretrained_model=outputs/temp/hf_my_model
"""

import sys
import os
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(__file__))

from evaluation.evaluator_chains_extended import (
    extract_input_vector, extract_output_vector,
)
from evaluation.correctness import check_completion_correct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ── Data loading ──

def load_unique_inputs(json_path):
    """Load training/test examples, deduplicate by input string."""
    with open(json_path) as f:
        raw = json.load(f)

    seen = {}
    examples = []
    for ex in raw:
        input_str = ex["input"]
        if input_str not in seen:
            input_vec = extract_input_vector(input_str)
            output_vec = extract_output_vector(input_str)
            chain_length = len(ex["letters"])
            seen[input_str] = True
            examples.append({
                "input_str": input_str,
                "input_vec": input_vec,
                "output_vec": output_vec,
                "chain_length": chain_length,
            })
    return examples


def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"
    return tokenizer


def load_model(model_path, requires_grad=True):
    log.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model.cuda()
    if not requires_grad:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def build_prompt_ids(input_str, tokenizer, split_str="[TRACE]"):
    text = f"{tokenizer.bos_token} {input_str} {split_str}"
    return tokenizer.encode(text, add_special_tokens=False)


# ── Generation ──

@torch.no_grad()
def generate_completions(model, tokenizer, batch_prompts, group_size,
                         temperature, max_length, forward_batch_size):
    """Generate group_size completions per prompt.

    Returns:
        texts: list[str] of length n_prompts * group_size (decoded after [TRACE])
        gen_ids: list[list[int]] of same length (full token sequences, PAD-stripped)
    """
    trace_id = tokenizer.encode("[TRACE]", add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Replicate each prompt group_size times
    expanded = []
    for prompt_ids in batch_prompts:
        for _ in range(group_size):
            expanded.append(prompt_ids)

    texts = []
    gen_ids = []

    model.eval()
    for b in range(0, len(expanded), forward_batch_size):
        batch_ids = expanded[b:b + forward_batch_size]
        batch_text = [tokenizer.decode(ids, skip_special_tokens=False)
                      for ids in batch_ids]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=pad_id,
            max_length=max_length,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            eos_token_id=eos_id,
        )

        for seq in outputs:
            seq_list = seq.tolist()

            # Strip leading PADs (from left-padded generation)
            while seq_list and seq_list[0] == pad_id:
                seq_list = seq_list[1:]
            gen_ids.append(seq_list)

            # Decode text after [TRACE]
            try:
                split_idx = seq_list.index(trace_id)
                remaining = seq_list[split_idx:]
                end_idx = (split_idx + remaining.index(eos_id)
                           if eos_id in remaining
                           else len(seq_list))
                pred = tokenizer.decode(
                    seq_list[split_idx + 1:end_idx], skip_special_tokens=True
                ).strip()
            except ValueError:
                pred = ""
            texts.append(pred)

    model.train()
    return texts, gen_ids


# ── Rewards and advantages ──

def compute_rewards(texts, batch_data, group_size):
    """Dense per-step rewards: fraction of decision steps that chose function f,
    gated on all blocks being arithmetically valid (invalid → 0).

    Returns:
        rewards: np.array (n_prompts, group_size)
        diagnostics: dict with completion-level counts
    """
    n_prompts = len(batch_data)
    n_total = n_prompts * group_size
    rewards = np.zeros((n_prompts, group_size))
    n_valid_fg = 0   # valid chain with any f/g decisions
    n_all_f = 0      # all decisions specifically from f
    n_valid_blocks = 0  # all blocks mathematically valid
    n_parseable = 0
    reward_sum = 0.0

    for i, pdata in enumerate(batch_data):
        for j in range(group_size):
            idx = i * group_size + j
            result = check_completion_correct(
                texts[idx], pdata["input_vec"], pdata["output_vec"]
            )
            if result["n_blocks"] > 0:
                n_parseable += 1
            if result["all_blocks_valid"]:
                n_valid_blocks += 1
            if result["correct"]:
                n_valid_fg += 1
            if result["all_chose_f"] and result["all_blocks_valid"]:
                n_all_f += 1

            # Dense reward: fraction of steps that chose f, gated on valid arithmetic
            chose_f = result["chose_f"]
            if result["all_blocks_valid"] and len(chose_f) > 0:
                r = sum(chose_f) / len(chose_f)
                rewards[i, j] = r
                reward_sum += r

    diagnostics = {
        "grpo/completions_parseable": n_parseable / max(n_total, 1),
        "grpo/completions_blocks_valid": n_valid_blocks / max(n_total, 1),
        "grpo/completions_valid_fg": n_valid_fg / max(n_total, 1),
        "grpo/completions_all_f": n_all_f / max(n_total, 1),
        "grpo/reward_mean_nonzero": reward_sum / max(n_valid_blocks, 1),
    }
    return rewards, diagnostics


def compute_group_advantages(rewards):
    """Group-normalized advantages (GRPO).

    For each group: A_j = (r_j - mean) / max(std, eps)
    Returns: np.array same shape as rewards
    """
    eps = 1e-8
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / np.maximum(std, eps)


# ── Sequence preparation for forward pass ──

def prepare_sequences(gen_ids_list, tokenizer):
    """Right-pad generated sequences for the forward pass.

    Returns:
        input_ids: (N, max_len) long tensor
        attention_mask: (N, max_len) long tensor
        response_starts: list[int] -- position of [TRACE] in each sequence
    """
    pad_id = tokenizer.pad_token_id
    trace_id = tokenizer.encode("[TRACE]", add_special_tokens=False)[0]
    N = len(gen_ids_list)
    max_len = max(len(ids) for ids in gen_ids_list)

    input_ids = torch.full((N, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(N, max_len, dtype=torch.long)
    response_starts = []

    for i, ids in enumerate(gen_ids_list):
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids)
        attention_mask[i, :seq_len] = 1
        try:
            response_starts.append(ids.index(trace_id))
        except ValueError:
            response_starts.append(seq_len - 1)

    return input_ids, attention_mask, response_starts


def make_response_mask(input_ids, attention_mask, response_starts):
    """Mask for all response tokens in shifted log-prob space (B, L-1).

    In shifted log-prob space, position j predicts token[j+1].
    We want gradient on all positions j >= response_start (i.e., all tokens
    after [TRACE]), within the valid (non-pad) region.
    """
    B, L = input_ids.shape
    mask = torch.zeros(B, L - 1, dtype=torch.float32)
    valid = attention_mask[:, 1:] > 0
    for i in range(B):
        mask[i, response_starts[i]:] = 1.0
    mask = mask * valid.float()
    return mask


def make_decision_mask(input_ids, attention_mask, tokenizer):
    """Mask for decision token positions only in shifted log-prob space (B, L-1).

    Decision tokens are the letter tokens (a-t) that appear right after
    [TRACE] or ';'.  These are the only positions where the model is making
    a choice — the rest of the trace is deterministic arithmetic.

    In shifted log-prob space, position j = log P(token[j+1] | token[:j+1]).
    A decision position is where input_ids[j] is [TRACE] or ';', meaning
    we are predicting the decision letter at position j+1.
    """
    trace_id = tokenizer.encode("[TRACE]", add_special_tokens=False)[0]
    semi_id = tokenizer.encode(";", add_special_tokens=False)[0]

    triggers = (input_ids[:, :-1] == trace_id) | (input_ids[:, :-1] == semi_id)
    valid = attention_mask[:, 1:] > 0
    mask = (triggers & valid).float()
    return mask


# ── Forward pass ──

def get_per_token_log_probs(model, input_ids, attention_mask):
    """Compute per-token log probs. Returns (B, L-1) tensor."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    return log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)


# ── GRPO loss ──

def grpo_loss(current_lps, old_lps, advantages, response_mask, clip_param):
    """Clipped policy gradient loss (per-token, sequence-level advantages).

    Returns: (loss, stats_dict)
    """
    ratio = torch.exp(current_lps - old_lps)
    adv = advantages.unsqueeze(1)  # (B, 1) broadcast over tokens

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
    token_loss = -torch.min(surr1, surr2)

    n_tokens = response_mask.sum().clamp(min=1)
    loss = (token_loss * response_mask).sum() / n_tokens

    with torch.no_grad():
        mean_ratio = (ratio * response_mask).sum() / n_tokens
        clipped = (
            ((ratio < 1 - clip_param) | (ratio > 1 + clip_param))
            & (response_mask > 0)
        )
        clipped_frac = clipped.float().sum() / n_tokens

    stats = {"ratio_mean": mean_ratio.item(), "clipped_frac": clipped_frac.item()}
    return loss, stats


# ── Training metrics ──

def compute_training_metrics(rewards, batch_data, group_size):
    """Compute detailed training metrics from rewards."""
    n_prompts = len(batch_data)
    metrics = {}

    metrics["grpo/reward_mean"] = float(rewards.mean())
    metrics["grpo/reward_std"] = float(rewards.std())

    # Group signal analysis
    all_correct = (rewards.sum(axis=1) == group_size).sum()
    all_wrong = (rewards.sum(axis=1) == 0).sum()
    mixed = n_prompts - all_correct - all_wrong

    metrics["grpo/groups_all_correct"] = int(all_correct)
    metrics["grpo/groups_all_wrong"] = int(all_wrong)
    metrics["grpo/groups_with_signal"] = int(mixed)
    metrics["grpo/signal_rate"] = float(mixed / max(n_prompts, 1))
    metrics["grpo/group_pass_rate"] = float((rewards.sum(axis=1) > 0).mean())

    # Per chain length
    len_groups = defaultdict(list)
    for i, pdata in enumerate(batch_data):
        len_groups[pdata["chain_length"]].append(i)

    for cl, indices in sorted(len_groups.items()):
        cl_rewards = rewards[indices]
        n_cl = len(indices)
        cl_mixed = (
            n_cl
            - (cl_rewards.sum(axis=1) == group_size).sum()
            - (cl_rewards.sum(axis=1) == 0).sum()
        )
        metrics[f"grpo/reward_mean_len{cl}"] = float(cl_rewards.mean())
        metrics[f"grpo/signal_rate_len{cl}"] = float(cl_mixed / max(n_cl, 1))
        metrics[f"grpo/group_pass_rate_len{cl}"] = float(
            (cl_rewards.sum(axis=1) > 0).mean()
        )

    return metrics


# ── Evaluation ──

@torch.no_grad()
def evaluate_greedy(model, tokenizer, eval_inputs, cfg):
    """Greedy generation on test inputs. Returns metrics dict."""
    trace_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    batch_size = cfg.eval.batch_size

    n = min(len(eval_inputs), cfg.eval.num_examples)
    rng = np.random.RandomState(cfg.grpo.seed)
    if n < len(eval_inputs):
        indices = rng.choice(len(eval_inputs), n, replace=False)
    else:
        indices = list(range(n))
    eval_data = [eval_inputs[i] for i in indices]

    prompts = [build_prompt_ids(ex["input_str"], tokenizer, cfg.data.split_str)
               for ex in eval_data]

    predictions = []
    model.eval()
    for b in range(0, len(prompts), batch_size):
        batch = prompts[b:b + batch_size]
        batch_text = [tokenizer.decode(ids, skip_special_tokens=False)
                      for ids in batch]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=pad_id,
            max_length=cfg.model.block_size,
            do_sample=False,
            eos_token_id=eos_id,
        )

        for seq in outputs:
            seq = seq.tolist()
            try:
                split_idx = seq.index(trace_id)
                end_idx = (seq.index(eos_id)
                           if eos_id in seq[split_idx:]
                           else len(seq))
                pred = tokenizer.decode(
                    seq[split_idx + 1:end_idx], skip_special_tokens=True
                ).strip()
            except ValueError:
                pred = ""
            predictions.append(pred)
    model.train()

    correct = 0
    correct_f = 0
    correct_per_len = defaultdict(int)
    correct_f_per_len = defaultdict(int)
    total_per_len = defaultdict(int)

    for i, ex in enumerate(eval_data):
        result = check_completion_correct(
            predictions[i], ex["input_vec"], ex["output_vec"]
        )
        total_per_len[ex["chain_length"]] += 1
        if result["correct"]:
            correct += 1
            correct_per_len[ex["chain_length"]] += 1
        if result["all_chose_f"] and result["all_blocks_valid"]:
            correct_f += 1
            correct_f_per_len[ex["chain_length"]] += 1

    n = max(len(eval_data), 1)
    metrics = {
        "eval/valid_solution_frac": correct / n,
        "eval/valid_solution_all_f_frac": correct_f / n,
    }
    for cl in sorted(total_per_len):
        d = max(total_per_len[cl], 1)
        metrics[f"eval/valid_solution_len{cl}"] = correct_per_len[cl] / d
        metrics[f"eval/valid_solution_all_f_len{cl}"] = correct_f_per_len[cl] / d
    return metrics


# ── Main ──

@hydra.main(config_path="config", config_name="base_grpo", version_base=None)
def main(cfg: DictConfig):
    # Setup
    torch.manual_seed(cfg.grpo.seed)
    np.random.seed(cfg.grpo.seed)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.proj_name, name=cfg.model.name, config=wandb_config
    )

    log.info(f"GRPO post-training: {cfg.model.name}")
    log.info(f"  Pretrained model: {cfg.grpo.pretrained_model}")
    log.info(f"  Group size: {cfg.grpo.group_size}, "
             f"Temperature: {cfg.grpo.temperature}")
    log.info(f"  Iterations: {cfg.grpo.n_iterations}, LR: {cfg.grpo.lr}")

    # Load model and tokenizer
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    model = load_model(cfg.grpo.pretrained_model, requires_grad=True)
    model.train()

    ref_model = None
    if cfg.grpo.kl_coeff > 0:
        ref_model = load_model(cfg.grpo.pretrained_model, requires_grad=False)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {total_params:,}")

    # Load data
    from hydra.utils import to_absolute_path
    train_inputs = load_unique_inputs(to_absolute_path(cfg.data.train_file))
    eval_inputs = load_unique_inputs(to_absolute_path(cfg.data.test_file))
    log.info(f"  Train inputs: {len(train_inputs)} unique")
    log.info(f"  Eval inputs: {len(eval_inputs)} unique")

    train_prompts = [
        build_prompt_ids(ex["input_str"], tokenizer, cfg.data.split_str)
        for ex in train_inputs
    ]

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.grpo.lr)

    def get_lr(iteration):
        if iteration < cfg.grpo.warmup_iters:
            return (iteration + 1) / cfg.grpo.warmup_iters
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Baseline evaluation
    log.info("Running baseline evaluation...")
    baseline = evaluate_greedy(model, tokenizer, eval_inputs, cfg)
    for k, v in sorted(baseline.items()):
        log.info(f"  Baseline {k}: {v:.4f}")
    wandb.log({
        "iteration": 0,
        **{f"baseline/{k.split('/')[-1]}": v for k, v in baseline.items()},
    })

    # GRPO loop
    K = cfg.grpo.group_size
    fbs = cfg.grpo.forward_batch_size
    rng = np.random.RandomState(cfg.grpo.seed)

    for iteration in range(1, cfg.grpo.n_iterations + 1):
        # 1. Sample batch of inputs
        batch_idx = rng.choice(
            len(train_inputs), cfg.grpo.inputs_per_batch, replace=False
        )
        batch_data = [train_inputs[i] for i in batch_idx]
        batch_prompts = [train_prompts[i] for i in batch_idx]

        # 2. Generate K completions per input
        texts, gen_ids = generate_completions(
            model, tokenizer, batch_prompts, K,
            cfg.grpo.temperature, cfg.model.block_size, fbs,
        )

        # 3. Compute rewards and advantages
        rewards, reward_diagnostics = compute_rewards(texts, batch_data, K)
        advantages = compute_group_advantages(rewards)

        # 4. Training metrics
        train_metrics = compute_training_metrics(rewards, batch_data, K)
        train_metrics.update(reward_diagnostics)

        # 5. Prepare sequences for forward pass
        input_ids, attention_mask, response_starts = prepare_sequences(
            gen_ids, tokenizer
        )
        resp_mask = make_decision_mask(input_ids, attention_mask, tokenizer)
        adv_flat = torch.tensor(
            advantages.flatten(), dtype=torch.float32
        ).cuda()
        n_sequences = input_ids.shape[0]
        n_response_tokens = resp_mask.sum().item()

        # 6. Compute old log probs (no grad, mini-batched)
        old_lps_list = []
        with torch.no_grad():
            for s in range(0, n_sequences, fbs):
                e = min(s + fbs, n_sequences)
                mb_lps = get_per_token_log_probs(
                    model, input_ids[s:e].cuda(), attention_mask[s:e].cuda()
                )
                old_lps_list.append((mb_lps * resp_mask[s:e].cuda()).cpu())
        old_token_lps = torch.cat(old_lps_list, dim=0)

        # Reference log probs (if KL penalty enabled)
        ref_token_lps = None
        if ref_model is not None:
            ref_list = []
            with torch.no_grad():
                for s in range(0, n_sequences, fbs):
                    e = min(s + fbs, n_sequences)
                    mb_lps = get_per_token_log_probs(
                        ref_model, input_ids[s:e].cuda(),
                        attention_mask[s:e].cuda()
                    )
                    ref_list.append((mb_lps * resp_mask[s:e].cuda()).cpu())
            ref_token_lps = torch.cat(ref_list, dim=0)

        # 7. Policy update (mini-epochs with gradient accumulation)
        total_loss = 0.0
        total_stats = defaultdict(float)

        for _epoch in range(cfg.grpo.mini_epochs):
            optimizer.zero_grad()
            epoch_loss = 0.0

            for s in range(0, n_sequences, fbs):
                e = min(s + fbs, n_sequences)
                mb_ids = input_ids[s:e].cuda()
                mb_attn = attention_mask[s:e].cuda()
                mb_resp = resp_mask[s:e].cuda()
                mb_old = old_token_lps[s:e].cuda()
                mb_adv = adv_flat[s:e]

                mb_cur = get_per_token_log_probs(model, mb_ids, mb_attn)
                mb_cur = mb_cur * mb_resp

                loss, stats = grpo_loss(
                    mb_cur, mb_old, mb_adv, mb_resp, cfg.grpo.clip_param
                )

                if ref_token_lps is not None:
                    mb_ref = ref_token_lps[s:e].cuda()
                    kl = ((mb_cur - mb_ref) * mb_resp).sum() / (
                        mb_resp.sum().clamp(min=1)
                    )
                    loss = loss + cfg.grpo.kl_coeff * kl
                    stats["kl"] = kl.item()

                # Scale for gradient accumulation
                mb_tokens = mb_resp.sum().item()
                scale = mb_tokens / max(n_response_tokens, 1)
                (loss * scale).backward()
                epoch_loss += loss.item() * scale

                for k, v in stats.items():
                    total_stats[k] += v * scale

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grpo.max_grad_norm
            )
            optimizer.step()
            total_loss += epoch_loss

        scheduler.step()
        total_loss /= max(cfg.grpo.mini_epochs, 1)

        # 8. Log metrics
        log_dict = {
            "iteration": iteration,
            "grpo/loss": total_loss,
            "grpo/lr": scheduler.get_last_lr()[0],
            "grpo/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "grpo/n_response_tokens": n_response_tokens,
            **train_metrics,
        }
        for k, v in total_stats.items():
            log_dict[f"grpo/{k}"] = v / max(cfg.grpo.mini_epochs, 1)
        wandb.log(log_dict)

        if iteration % 10 == 0 or iteration == 1:
            log.info(
                f"Iter {iteration}/{cfg.grpo.n_iterations} | "
                f"loss={total_loss:.2e} | "
                f"grad_norm={log_dict['grpo/grad_norm']:.4f} | "
                f"reward={train_metrics['grpo/reward_mean']:.3f} | "
                f"signal={train_metrics['grpo/signal_rate']:.2f}"
            )

        # 9. Periodic evaluation
        if iteration % cfg.grpo.eval_every == 0:
            eval_metrics = evaluate_greedy(model, tokenizer, eval_inputs, cfg)
            wandb.log({"iteration": iteration, **eval_metrics})
            parts = [f"  Eval iter {iteration}:"]
            for k, v in sorted(eval_metrics.items()):
                parts.append(f"{k}={v:.4f}")
            log.info(" | ".join(parts))

        # 10. Periodic save
        if iteration % cfg.grpo.save_every == 0:
            ckpt_dir = Path(cfg.output.checkpoint_dir) / f"iter_{iteration}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            log.info(f"  Checkpoint saved to {ckpt_dir}")

    # Final save
    out_dir = Path(cfg.output.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(f"Final model saved to {out_dir}")

    # Final evaluation
    final = evaluate_greedy(model, tokenizer, eval_inputs, cfg)
    wandb.log({"iteration": cfg.grpo.n_iterations, **final})

    log.info("\n=== GRPO Training Complete ===")
    log.info(f"  Baseline pass@1: "
             f"{baseline.get('eval/valid_solution_frac', 0):.4f}")
    log.info(f"  Final pass@1:    "
             f"{final.get('eval/valid_solution_frac', 0):.4f}")
    for cl in [3, 4, 5]:
        b = baseline.get(f"eval/valid_solution_len{cl}", 0)
        f_ = final.get(f"eval/valid_solution_len{cl}", 0)
        log.info(f"  Length {cl}: {b:.4f} -> {f_:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
