# LoRA / PEFT for Qwen3-8B Math — Best-Practice Review

## Overview

At the 40 GB A100 / Qwen3-8B scale, LoRA is the default path. Full fine-tuning of the
8.2 B dense weights (~16.4 GB in bf16) does not fit alongside AdamW optimizer state
(~33 GB for full parameters) and activations; LoRA reduces the trainable parameter
count by 100–1000× and the optimizer footprint by a similar factor, freeing headroom
for activations, longer sequences, and larger micro-batches. Recent work (Thinking
Machines Lab, *LoRA Without Regret*, 2025) shows that well-tuned LoRA, targeted at
**all** linear layers including MLP, **matches full-FT loss curves on GSM8K and MATH**
once rank is sufficient (~256–512 for dense SFT; as low as rank 1 suffices for RL
policy-gradient updates). This directly invalidates the older "LoRA < full FT on
reasoning" narrative that motivated the naive_ft baselines in the sibling repo.

## Core method recap

- **LoRA** (Hu et al. 2021): freeze base weights W, learn a low-rank update ΔW = BA
  with B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d, k). Effective update scales by α/r.
  Trainable-param count ≈ r · (d_in + d_out) per targeted linear layer.
- **DoRA** (Liu et al. ICML 2024 Oral, arXiv:2402.09353): decomposes each pretrained
  weight W into magnitude m (learned scalar per output dim) and direction V/||V||,
  applies LoRA only to the directional component. Reported gains of +0.7 avg over
  LoRA across commonsense reasoning suites on LLaMA-7B/13B/2-7B/3-8B; also wins on
  VL-BART and LLaVA. Cost: ~2× forward compute vs LoRA at same rank, but often
  reaches LoRA's accuracy with half the rank.
- **QLoRA** (Dettmers et al. NeurIPS 2023, arXiv:2305.14314): 4-bit NF4 quantization
  of base weights + double quantization + paged optimizer. Pushes 7 B weights from
  ~14 GB to ~3.5 GB. NF4 with double-quant recovers 16-bit LoRA MMLU exactly; FP4
  lags ~1 pt.
- **LoftQ** (Li et al. ICLR 2024, arXiv:2310.08659): joint quantization + LoRA
  initialization — seeks Q and BA such that Q + BA ≈ W_fp16, instead of QLoRA's
  `Q=quant(W), B=0, A~Gaussian`. At 4-bit LoftQ matches QLoRA; at 2-bit it beats
  QLoRA by 8–10 pt on MNLI/SQuAD where QLoRA collapses.

## Hyperparameters — what works on math (with citations)

| Knob | Recommended | Notes |
|------|-------------|-------|
| Target modules | all linear: `q,k,v,o,gate,up,down` | *LoRA Without Regret*: attention-only significantly underperforms MLP-only at matched params, on both Llama-3.1-8B and Qwen3-30B-A3B. LLaMA-Factory default `target=all`; Unsloth same. |
| Rank r | **32–64 for SFT on math**; 16 viable | Databricks/Unsloth suggest 16/32; Thinking Machines shows rank-1 matches FullFT for RL, but SFT benefits from 128–512 when dataset is large. NeuroProlog: rank 48 for 8 B math SFT. |
| Alpha α | α = r (standard) or 2r (aggressive) | Effective scale α/r. Most math recipes use 2r. |
| Dropout | 0.0–0.1 | LLaMA-Factory default 0.1; Unsloth 0.0 for LoRA, 0.05 for reasoning tasks. |
| LR | **2e-4 to 3e-4** | Unsloth default 2e-4. Thinking Machines rule-of-thumb: optimal LoRA LR ≈ **10× full-FT LR** (short runs up to 15×). Qwen2.5-Math full-FT used 7e-6 → 7e-7; LoRA equivalent ≈ 7e-5 to 7e-4. |
| Warmup ratio | **3–10%** (5% typical) | Critical for LoRA stability — under-warmed runs diverge early. Linear warmup → cosine decay. |
| Epochs | **2–3** | >3 epochs degrades on instruction/math SFT (Unsloth, LLaMA-Factory consensus). |
| Effective batch | 16–64 | Unsloth: per-device 2 × grad-accum 8 = 16. Scale per VRAM. |
| Grad clip | 1.0 | Standard. |
| Weight decay | 0.0–0.1 | Qwen2.5-Math used 0.1 full-FT; LoRA tolerates 0.0. |
| Precision | **bf16** | A100 native; flash-attention-2 in bf16 is fastest path. |
| Grad checkpoint | **on** for 8 B on 40 GB at seq 2k+ | Trades ~20% step time for ~40% activation-memory reduction; essential when bs·seq > ~8k tokens. |
| Sequence length | 2 k (MATH solutions typically <1.5 k) | Qwen3 RoPE base scales to 32 k native / 131 k via YaRN; no issue at 2 k. Unsloth Qwen3 tutorial also recommends 2 k for testing. |
| Optimizer | AdamW (or paged_adamw_8bit) | 8-bit paged optimizer saves ~2 GB at negligible accuracy cost. |

### Learning rate schedule gotcha

LoRA is especially sensitive to LR decay: vanilla LoRA at the *right* LR equals full FT,
but at the wrong LR it underperforms dramatically (Shuttleworth et al. 2026, *Learning
Rate Matters*). Use cosine decay to ~10% of peak and do not stop warmup early.

### Rank-vs-data interaction

"LoRA Learns Less and Forgets Less" (Biderman et al. 2024): when rank >> intrinsic
task rank, extra capacity is wasted and can worsen base-task retention. For our 1 k-
sample MATH subset, rank 16–32 should saturate. If training on the full 12 k MATH or
large GSM8K variants, push to 64–128.

## Qwen3-specific considerations

- **Architecture** (Qwen3 Technical Report, 2025): 36 layers, hidden 4096, 32 Q-heads
  / **8 KV-heads** (GQA with ratio 4:1), SwiGLU MLP (so MLP has `gate_proj`,
  `up_proj`, `down_proj`), RMSNorm pre-norm, QK-Norm for stability, RoPE base
  extended to 1 M (YaRN scaling to 131 k context).
- **GQA implication for LoRA**: `k_proj` and `v_proj` are 4× narrower than `q_proj`
  (1024 vs 4096). A rank-64 LoRA on `k_proj` is ~131 k params vs 524 k on `q_proj`.
  If budgeting strictly by param count, the per-module rank can be uneven; in
  practice keep r uniform — extra capacity on `q_proj`/MLP is where reasoning gains
  come from (*LoRA Without Regret*).
- **Tied embeddings**: Qwen3 dense models share input/output embeddings. Do NOT add
  LoRA to `embed_tokens`/`lm_head` unless explicitly unfreezing the tied head; it
  breaks tying and costs ~1 GB extra.
- **Thinking-mode supervision**: Qwen3 emits `<think>...</think>` blocks before the
  final answer. For math SFT the common decisions are:
  (a) **Supervise both think and answer** (default Unsloth Qwen3 SFT). Risk:
  overfitting on a specific thinking style; may hurt zero-shot thinking on held-out
  problems.
  (b) **Supervise answer only, mask think block** via `loss_mask`. Cleaner, but
  requires custom collator. Preferred when the training traces are ours (no GT
  thinking).
  (c) Use `enable_thinking=False` and treat as non-thinking SFT. Loses reasoning
  capability, not recommended for MATH.
  Given MATH has GT LaTeX CoT in the `solution` field, option (a) with an explicit
  `<think>solution</think>\boxed{answer}` template is the natural choice.
- **Chat template**: Qwen3 uses `<|im_start|>role\n...<|im_end|>`. Always mask loss
  on system/user spans; apply `apply_chat_template(enable_thinking=True)` to keep
  training/inference prompts aligned.

## VRAM budget for 40 GB A100 training of Qwen3-8B

Rough accounting (bf16 base, LoRA r=32 all-linear, seq 2048, micro-batch 2):

| Component | Size | Notes |
|-----------|------|-------|
| Base weights (bf16) | 16.4 GB | 8.19 B × 2 B |
| LoRA adapters + grads | ~0.15 GB | ~40 M trainable params × 2 (param + grad) × 2 B (bf16) — negligible |
| AdamW state (on LoRA only) | ~0.3 GB | 2 states × 4 B × 40 M params |
| Activations (seq 2 k, bs 2, **no** ckpt) | ~18–22 GB | 36 layers × roughly 0.25 GB/layer/batch at seq 2k in bf16 |
| Activations (seq 2 k, bs 2, **with** ckpt) | ~6–8 GB | √L reduction |
| KV cache (training) | included in activations | |
| Flash-attn workspace, CUDA overhead | ~1–2 GB | |
| **Total no-ckpt** | ~37 GB | tight; OOM risk with any bs increase |
| **Total with grad-ckpt** | ~24–26 GB | comfortable; room for bs up to 4 or seq 4 k |

**Recommendation**: enable `gradient_checkpointing=True` from the start. At bs=2 and
seq=2048 you have ~15 GB headroom — use it for grad accumulation to reach effective
batch 16–32 rather than chasing bigger micro-batch. If moving to Qwen3-14B, you'll
need **QLoRA 4-bit** to fit (weights drop from ~29 → ~8 GB).

### Is QLoRA worth it for Qwen3-8B on 40 GB?

Probably **no** for us. bf16 fits, QLoRA costs ~1 pt on some tasks (FP4) or is
exactly equal (NF4), and 4-bit dequant kernels slow the forward pass ~20%. Keep
QLoRA in the back pocket for: (1) Qwen3-14B comparison; (2) if we need seq 8 k+;
(3) if we start training on multi-GPU and bandwidth becomes the bottleneck.

## Libraries

- **HF PEFT** — lingua franca; cleanest for research code. Direct `LoraConfig`
  control. Use `target_modules="all-linear"` in PEFT ≥0.10 for automatic coverage.
- **LLaMA-Factory** — YAML-driven, rich recipe library, supports DoRA/QLoRA/LoftQ
  flags. Defaults: r=8, α=16, dropout=0.1, target=all.
- **Unsloth** — 2× speedup via fused kernels; Triton-based. Best choice if we want
  to iterate fast on a single GPU. Its Qwen3 tutorial is current and its chat-
  template handling for thinking mode is battle-tested.
- **axolotl** — heavier but production-grade; nice for multi-GPU later.

For this project (single A100, research workflow, need tight integration with our
memory module), stay on **HF PEFT + transformers Trainer** for minimal magic, with
the option to swap in Unsloth's optimized Qwen3 kernels later for speed.

## Comparison with external memory module (our prior work)

The sibling `memory_experiment/` attaches a `MemoryAttention` module (W_q + small KV
bank) to a frozen transformer; `naive_ft/` was the paired LoRA/full-FT baseline on
the same synthetic task. On Qwen3-8B at hidden=4096, a naïve W_q alone is ~16 M
params (vs ~262 k in the 12L-512D toy). Key expected differences at matched
trainable-param count (say, ~40 M):

1. **Coverage**: LoRA r=32 on all 36 layers × 7 projections is uniform across depth;
   memory module lives at one layer and has to be read through subsequent frozen
   layers. For math reasoning (long-range compositional), LoRA's depth-uniform
   updates probably win.
2. **Bias profile**: Our memory module acts like a conditional residual *at one
   layer* and is great for "insert a learned fact"-type corrections. LoRA is a
   distributed perturbation of the forward function. On MATH, we expect LoRA to
   improve the whole reasoning chain; memory to improve specific symbolic
   substitutions (e.g. definitions, common sub-problems).
3. **Forgetting**: Memory is more additive (weights untouched outside the hook
   path) → typically lower forgetting of base capabilities. LoRA with excessive
   rank or wrong LR can drift (see Biderman 2024).
4. **Decision-point control**: Memory gives us a direct, interpretable read/write
   surface at the hook layer; LoRA does not. This is the main reason to keep the
   memory track alive even if LoRA wins on headline MATH accuracy.

The prior naive_ft results on the toy task showed LoRA was roughly parity with full
FT and memory beat both on the narrow alignment sub-task (mode-A ffff data). On
natural-language MATH this ordering is unlikely to hold — LoRA is probably the
stronger baseline, memory remains the differentiated method for specific
compositional/alignment questions.

## Actionable recipe for Phase 2

Starting config (HF PEFT + transformers Trainer, Qwen3-8B bf16 on 40 GB A100):

```python
LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
# Training:
# lr=2e-4, warmup_ratio=0.05, scheduler=cosine, min_lr_ratio=0.1
# weight_decay=0.0, grad_clip=1.0, epochs=3
# per_device_bs=2, grad_accum=8  (effective 16)
# bf16=True, gradient_checkpointing=True, flash_attention_2=True
# max_seq_len=2048
# Loss: SFT on full assistant turn including <think> block (MATH has GT CoT)
```

Rationale: r=32 α=64 is a safe middle ground (beats r=8 on math in Databricks
studies, doesn't overfit 1 k-sample MATH like r=128 might); LR 2e-4 is Unsloth's
tested default and sits inside the 7e-5 – 7e-4 band implied by Qwen2.5-Math's
full-FT LR × 10; 3 epochs respects the "diminishing returns after 3" rule; grad
checkpoint keeps us at ~25 GB VRAM with 15 GB margin for seq-length experiments.
First ablation to run: rank sweep {16, 32, 64} holding α/r=2 fixed, to locate the
saturation point for our MATH+GSM8K subset. Second ablation: attention-only vs
all-linear, to re-confirm the MLP-matters-for-reasoning finding on our exact
dataset.

## Sources

- Hu et al., *LoRA: Low-Rank Adaptation of LLMs* (2021) — https://arxiv.org/abs/2106.09685
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024 Oral — https://arxiv.org/abs/2402.09353 ; https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023 — https://arxiv.org/abs/2305.14314
- Li et al., *LoftQ: LoRA-Fine-Tuning-Aware Quantization*, ICLR 2024 — https://arxiv.org/abs/2310.08659 ; https://www.microsoft.com/en-us/research/blog/loftq-reimagining-llm-fine-tuning-with-smarter-initialization/
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024) — https://arxiv.org/abs/2405.09673
- Thinking Machines Lab, *LoRA Without Regret* (2025) — https://thinkingmachines.ai/blog/lora/
- *Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning* — https://arxiv.org/pdf/2602.04998
- Qwen3 Technical Report (2025) — https://arxiv.org/abs/2505.09388
- Qwen2.5-Math Technical Report — https://arxiv.org/abs/2409.12122
- Unsloth Qwen3 fine-tune tutorial — https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
- Unsloth LoRA hyperparameters guide — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- LLaMA-Factory tuning algorithms — https://llamafactory.readthedocs.io/en/latest/advanced/tuning_algorithms.html
- HF PEFT LoRA reference — https://huggingface.co/docs/peft/en/package_reference/lora
- Databricks, *Efficient Fine-Tuning with LoRA* — https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
- Qwen/Qwen3-8B model card — https://huggingface.co/Qwen/Qwen3-8B
- OPLoRA orthogonal-projection forgetting mitigation — https://arxiv.org/html/2510.13003
- NeuroProlog multi-task math fine-tune — https://arxiv.org/html/2603.02504
