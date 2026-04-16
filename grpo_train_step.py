"""
GRPO with per-step advantage decomposition for decision chain models.

Unlike standard GRPO (sequence-level advantages), this computes independent
advantages for each decision point in the chain. At step k, the advantage
is based on whether THIS step chose f, normalized across the K completions.

This solves the credit assignment problem: each decision position gets a
direct gradient toward f, independent of other decisions in the chain.

Usage:
    python grpo_train_step.py
    python grpo_train_step.py grpo.lr=1e-5 grpo.kl_coeff=0.05
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

from evaluation.correctness import check_completion_correct
from grpo_train import (
    load_unique_inputs, load_tokenizer, load_model, build_prompt_ids,
    generate_completions, prepare_sequences, make_decision_mask,
    make_response_mask, get_per_token_log_probs, evaluate_greedy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MAX_CHAIN_STEPS = 5


# ── Per-step rewards ──

def compute_step_rewards(texts, batch_data, group_size):
    """Per-step binary rewards: 1 if step chose f, 0 otherwise.

    Only assigns rewards for completions where ALL blocks are valid
    (ensures intermediate vectors are trustworthy for chose_f checks).

    Returns:
        step_chose_f: np.array (n_prompts, group_size, MAX_CHAIN_STEPS)
        step_valid: np.array (n_prompts, group_size, MAX_CHAIN_STEPS)
        diagnostics: dict
    """
    n_prompts = len(batch_data)
    n_total = n_prompts * group_size

    step_chose_f = np.zeros((n_prompts, group_size, MAX_CHAIN_STEPS))
    step_valid = np.zeros((n_prompts, group_size, MAX_CHAIN_STEPS))

    n_parseable = 0
    n_valid_blocks = 0
    n_valid_fg = 0
    n_all_f = 0
    total_valid_steps = 0
    total_f_steps = 0

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

            # Per-step: only for completions with valid arithmetic
            if result["all_blocks_valid"] and len(result["chose_f"]) > 0:
                for k, cf in enumerate(result["chose_f"]):
                    if k < MAX_CHAIN_STEPS:
                        step_chose_f[i, j, k] = float(cf)
                        step_valid[i, j, k] = 1.0
                        total_valid_steps += 1
                        if cf:
                            total_f_steps += 1

    diagnostics = {
        "grpo/completions_parseable": n_parseable / max(n_total, 1),
        "grpo/completions_blocks_valid": n_valid_blocks / max(n_total, 1),
        "grpo/completions_valid_fg": n_valid_fg / max(n_total, 1),
        "grpo/completions_all_f": n_all_f / max(n_total, 1),
        "grpo/step_f_rate": total_f_steps / max(total_valid_steps, 1),
    }
    return step_chose_f, step_valid, diagnostics


# ── Per-step advantages ──

def compute_step_advantages(step_chose_f, step_valid):
    """Per-step group-normalized advantages.

    For each prompt i and step k, normalizes chose_f across the K
    completions where that step is valid. Steps where all completions
    agree (all chose f or all chose g) get zero advantage.

    Returns: np.array (n_prompts, group_size, MAX_CHAIN_STEPS)
    """
    eps = 1e-8
    n_prompts, group_size, max_steps = step_chose_f.shape
    advantages = np.zeros_like(step_chose_f)

    for i in range(n_prompts):
        for k in range(max_steps):
            valid_mask = step_valid[i, :, k] > 0
            n_valid = valid_mask.sum()
            if n_valid < 2:
                continue

            rewards_k = step_chose_f[i, valid_mask, k]
            mean = rewards_k.mean()
            std = rewards_k.std()
            if std < eps:
                continue  # all same — no signal at this step

            normalized = (step_chose_f[i, :, k] - mean) / std
            advantages[i, :, k] = normalized * step_valid[i, :, k]

    return advantages


def build_step_advantage_tensor(step_advantages_flat, gen_ids_list, tokenizer):
    """Map step-level advantages to token positions in shifted log-prob space.

    In shifted space, position j predicts token[j+1]. A decision position
    is where input_ids[j] is [TRACE] or ';' (predicting the letter at j+1).

    The k-th such position in a sequence gets step_advantages[k].

    Args:
        step_advantages_flat: np.array (B, MAX_CHAIN_STEPS)
        gen_ids_list: list of token id lists, length B
        tokenizer: tokenizer for [TRACE] and ';' lookup

    Returns: torch.Tensor (B, max_len - 1)
    """
    trace_id = tokenizer.encode("[TRACE]", add_special_tokens=False)[0]
    semi_id = tokenizer.encode(";", add_special_tokens=False)[0]

    B = len(gen_ids_list)
    max_len = max(len(ids) for ids in gen_ids_list)
    adv_tensor = torch.zeros(B, max_len - 1, dtype=torch.float32)

    for i, ids in enumerate(gen_ids_list):
        step = 0
        for j in range(len(ids) - 1):
            if ids[j] == trace_id or ids[j] == semi_id:
                if step < MAX_CHAIN_STEPS:
                    adv_tensor[i, j] = step_advantages_flat[i, step]
                    step += 1

    return adv_tensor


# ── Per-step GRPO loss ──

def grpo_loss_step(current_lps, old_lps, advantages_2d, response_mask, clip_param):
    """Clipped policy gradient loss with per-position advantages.

    Args:
        current_lps: (B, L-1) current log probs (masked)
        old_lps: (B, L-1) old log probs (masked)
        advantages_2d: (B, L-1) per-position advantages
        response_mask: (B, L-1) decision mask
        clip_param: PPO clipping epsilon

    Returns: (loss, stats_dict)
    """
    ratio = torch.exp(current_lps - old_lps)

    surr1 = ratio * advantages_2d
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_2d
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

def compute_step_metrics(step_chose_f, step_valid, batch_data, group_size):
    """Per-step training metrics."""
    n_prompts = len(batch_data)
    metrics = {}

    # Per-step signal analysis
    total_step_slots = 0
    signal_step_slots = 0
    f_count_total = 0
    valid_count_total = 0

    for i in range(n_prompts):
        for k in range(MAX_CHAIN_STEPS):
            valid_mask = step_valid[i, :, k] > 0
            n_valid = valid_mask.sum()
            if n_valid < 2:
                continue
            total_step_slots += 1
            f_count = step_chose_f[i, valid_mask, k].sum()
            valid_count_total += n_valid
            f_count_total += f_count
            if 0 < f_count < n_valid:
                signal_step_slots += 1

    metrics["grpo/step_signal_rate"] = signal_step_slots / max(total_step_slots, 1)
    metrics["grpo/step_f_mean"] = f_count_total / max(valid_count_total, 1)

    # Sequence-level: all-f rate, reward mean (for comparison with old approach)
    all_f_count = 0
    for i in range(n_prompts):
        for j in range(group_size):
            # Check if all valid steps chose f
            valid_steps = step_valid[i, j, :] > 0
            if valid_steps.sum() == 0:
                continue
            if step_chose_f[i, j, valid_steps].all():
                all_f_count += 1
    metrics["grpo/completions_all_f_seq"] = all_f_count / max(n_prompts * group_size, 1)

    # Per chain length
    len_groups = defaultdict(list)
    for i, pdata in enumerate(batch_data):
        len_groups[pdata["chain_length"]].append(i)

    for cl, indices in sorted(len_groups.items()):
        cl_total = 0
        cl_signal = 0
        cl_f = 0
        cl_valid = 0
        for i in indices:
            for k in range(cl):
                vm = step_valid[i, :, k] > 0
                nv = vm.sum()
                if nv < 2:
                    continue
                cl_total += 1
                fc = step_chose_f[i, vm, k].sum()
                cl_valid += nv
                cl_f += fc
                if 0 < fc < nv:
                    cl_signal += 1
        metrics[f"grpo/step_signal_rate_len{cl}"] = cl_signal / max(cl_total, 1)
        metrics[f"grpo/step_f_mean_len{cl}"] = cl_f / max(cl_valid, 1)

    return metrics


# ── Main ──

@hydra.main(config_path="config", config_name="base_grpo", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.grpo.seed)
    np.random.seed(cfg.grpo.seed)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.proj_name, name=cfg.model.name, config=wandb_config
    )

    log.info(f"GRPO per-step: {cfg.model.name}")
    log.info(f"  Pretrained model: {cfg.grpo.pretrained_model}")
    log.info(f"  Group size: {cfg.grpo.group_size}, "
             f"Temperature: {cfg.grpo.temperature}")
    log.info(f"  Iterations: {cfg.grpo.n_iterations}, LR: {cfg.grpo.lr}")

    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    model = load_model(cfg.grpo.pretrained_model, requires_grad=True)
    model.train()

    ref_model = None
    if cfg.grpo.kl_coeff > 0:
        ref_model = load_model(cfg.grpo.pretrained_model, requires_grad=False)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {total_params:,}")

    from hydra.utils import to_absolute_path
    train_inputs = load_unique_inputs(to_absolute_path(cfg.data.train_file))
    eval_inputs = load_unique_inputs(to_absolute_path(cfg.data.test_file))
    log.info(f"  Train inputs: {len(train_inputs)} unique")
    log.info(f"  Eval inputs: {len(eval_inputs)} unique")

    train_prompts = [
        build_prompt_ids(ex["input_str"], tokenizer, cfg.data.split_str)
        for ex in train_inputs
    ]

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

        # 3. Per-step rewards and advantages
        step_chose_f, step_valid, reward_diagnostics = compute_step_rewards(
            texts, batch_data, K
        )
        step_advantages = compute_step_advantages(step_chose_f, step_valid)

        # 4. Training metrics
        train_metrics = compute_step_metrics(
            step_chose_f, step_valid, batch_data, K
        )
        train_metrics.update(reward_diagnostics)

        # 5. Build advantage tensor mapped to token positions
        step_adv_flat = step_advantages.reshape(-1, MAX_CHAIN_STEPS)
        adv_tensor = build_step_advantage_tensor(
            step_adv_flat, gen_ids, tokenizer
        )

        # 6. Prepare sequences for forward pass
        input_ids, attention_mask, response_starts = prepare_sequences(
            gen_ids, tokenizer
        )
        decision_mask = make_decision_mask(input_ids, attention_mask, tokenizer)
        response_mask = make_response_mask(input_ids, attention_mask, response_starts)
        n_sequences = input_ids.shape[0]
        n_decision_tokens = decision_mask.sum().item()
        n_response_tokens = response_mask.sum().item()

        # 7. Compute old log probs on decision tokens (for policy ratio)
        old_lps_list = []
        with torch.no_grad():
            for s in range(0, n_sequences, fbs):
                e = min(s + fbs, n_sequences)
                mb_lps = get_per_token_log_probs(
                    model, input_ids[s:e].cuda(), attention_mask[s:e].cuda()
                )
                old_lps_list.append((mb_lps * decision_mask[s:e].cuda()).cpu())
        old_token_lps = torch.cat(old_lps_list, dim=0)

        # Reference log probs on ALL response tokens (for KL anchoring arithmetic)
        ref_response_lps = None
        if ref_model is not None:
            ref_list = []
            with torch.no_grad():
                for s in range(0, n_sequences, fbs):
                    e = min(s + fbs, n_sequences)
                    mb_lps = get_per_token_log_probs(
                        ref_model, input_ids[s:e].cuda(),
                        attention_mask[s:e].cuda()
                    )
                    ref_list.append((mb_lps * response_mask[s:e].cuda()).cpu())
            ref_response_lps = torch.cat(ref_list, dim=0)

        # 8. Policy update (hybrid mask: PG on decision, KL on all response)
        total_loss = 0.0
        total_stats = defaultdict(float)

        for _epoch in range(cfg.grpo.mini_epochs):
            optimizer.zero_grad()
            epoch_loss = 0.0

            for s in range(0, n_sequences, fbs):
                e = min(s + fbs, n_sequences)
                mb_ids = input_ids[s:e].cuda()
                mb_attn = attention_mask[s:e].cuda()
                mb_dec = decision_mask[s:e].cuda()
                mb_resp = response_mask[s:e].cuda()
                mb_old = old_token_lps[s:e].cuda()
                mb_adv = adv_tensor[s:e].cuda()

                # Full log probs (unmasked) — needed for both PG and KL
                mb_cur_full = get_per_token_log_probs(model, mb_ids, mb_attn)

                # Policy gradient on decision tokens only
                mb_cur_dec = mb_cur_full * mb_dec
                loss_pg, stats = grpo_loss_step(
                    mb_cur_dec, mb_old, mb_adv, mb_dec, cfg.grpo.clip_param
                )

                # KL on ALL response tokens (anchors arithmetic to reference)
                loss = loss_pg
                if ref_response_lps is not None:
                    mb_ref = ref_response_lps[s:e].cuda()
                    mb_cur_resp = mb_cur_full * mb_resp
                    kl = ((mb_cur_resp - mb_ref) * mb_resp).sum() / (
                        mb_resp.sum().clamp(min=1)
                    )
                    loss = loss + cfg.grpo.kl_coeff * kl
                    stats["kl"] = kl.item()

                # Scale by decision token fraction for gradient accumulation
                dec_scale = mb_dec.sum().item() / max(n_decision_tokens, 1)
                (loss * dec_scale).backward()
                epoch_loss += loss.item() * dec_scale

                for k, v in stats.items():
                    total_stats[k] += v * dec_scale

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grpo.max_grad_norm
            )
            optimizer.step()
            total_loss += epoch_loss

        scheduler.step()
        total_loss /= max(cfg.grpo.mini_epochs, 1)

        # 9. Log metrics
        log_dict = {
            "iteration": iteration,
            "grpo/loss": total_loss,
            "grpo/lr": scheduler.get_last_lr()[0],
            "grpo/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "grpo/n_decision_tokens": n_decision_tokens,
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
                f"grad={log_dict['grpo/grad_norm']:.4f} | "
                f"step_f={train_metrics['grpo/step_f_mean']:.3f} | "
                f"step_sig={train_metrics['grpo/step_signal_rate']:.2f} | "
                f"blk={train_metrics.get('grpo/completions_blocks_valid', 0):.3f}"
            )

        # 10. Periodic evaluation
        if iteration % cfg.grpo.eval_every == 0:
            eval_metrics = evaluate_greedy(model, tokenizer, eval_inputs, cfg)
            wandb.log({"iteration": iteration, **eval_metrics})
            parts = [f"  Eval iter {iteration}:"]
            for k, v in sorted(eval_metrics.items()):
                parts.append(f"{k}={v:.4f}")
            log.info(" | ".join(parts))

        # 11. Periodic save
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

    log.info("\n=== GRPO Per-Step Training Complete ===")
    log.info(f"  Baseline all-f: "
             f"{baseline.get('eval/valid_solution_all_f_frac', 0):.4f}")
    log.info(f"  Final all-f:    "
             f"{final.get('eval/valid_solution_all_f_frac', 0):.4f}")
    log.info(f"  Baseline pass@1: "
             f"{baseline.get('eval/valid_solution_frac', 0):.4f}")
    log.info(f"  Final pass@1:    "
             f"{final.get('eval/valid_solution_frac', 0):.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
