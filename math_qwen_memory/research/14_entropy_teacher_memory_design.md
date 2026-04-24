# Entropy-Gated Teacher → Memory Distillation — Design Doc

## One-line thesis

At each **high-entropy branching token** in a Qwen3-8B CoT, query a **stronger teacher** (Claude, GPT-4-class, or Qwen3-72B) for "advice", then **distill** that advice into a **trainable external KV memory module** that is hooked into the frozen backbone's residual stream and fires at similar hidden states at inference — no teacher needed at test time.

## Why this is worth a paper

| Ingredient | Novelty | Best prior |
|-----------|---------|-----------|
| Entropy-gated token selection | ❌ established | Wang 2025 "Beyond 80/20"; top-20% entropy acts as "fork tokens" |
| Teacher distillation | ❌ established | Standard KD from 2014; EOPD/TIP (2026) are entropy-gated variants |
| Frozen backbone + trainable memory | ❌ established | PKM-ResM (Kim 2020), LongMem (Wang 2023), Engram (DeepSeek Jan 2026) |
| **Combination**: entropy-triggered teacher query → distilled into KV memory on frozen backbone | ✅ not found as a single paper after literature search | Closest: EOPD (entropy+teacher, weights-level) and Engram (memory module, no teacher) |

The combination isolates an empirical question nobody has directly answered: **does a frozen-backbone external memory module match or beat LoRA/full-weight distillation when both are fed the same entropy-triggered teacher signal?** If yes, you get capability transfer without catastrophic forgetting. If no, the memory framing is aesthetic and you should use LoRA.

## Design — overview

```
                                 TRAIN
┌─────────────────────────────────────────────────────────────┐
│  Problem p ──► Qwen3-8B (frozen, thinking mode)             │
│                    │                                        │
│                    ▼  generates CoT token by token          │
│              per-token H(p_t) = entropy of next-token dist  │
│                    │                                        │
│                    ▼  threshold: H > τ (top-20%)            │
│             selected positions t1, t2, ... tk               │
│                    │                                        │
│                    ▼  for each ti:                          │
│                   hidden state h_ti ∈ R^4096 (from layer L) │
│                   prompt prefix s_ti                        │
│                    │                                        │
│                    ▼  query TEACHER with (problem, prefix)  │
│                   advice a_ti = teacher's continuation      │
│                    │                                        │
│                    ▼  convert a_ti → target vector v_ti     │
│                    │                                        │
│                    ▼  train memory bank (K, V):             │
│                    │    loss = KL(student | teacher) at ti  │
│                    │    gradient flows only through memory  │
└─────────────────────────────────────────────────────────────┘

                                 INFERENCE
┌─────────────────────────────────────────────────────────────┐
│  Problem p ──► Qwen3-8B (frozen) + memory hook              │
│                    │                                        │
│                    ▼  at every token, compute               │
│                   q = W_q · h_t                             │
│                   attn = softmax(q K^T / √d)                │
│                   h'_t = h_t + γ · attn · V                 │
│                    │                                        │
│                    ▼  next token from modified h'_t         │
│   No teacher call at inference.                             │
└─────────────────────────────────────────────────────────────┘
```

## Core design choices (most important)

### C1. What is "advice" from the teacher?

Four options, ranked by expected tractability:

| Option | What teacher returns | Memory value representation | Tradeoffs |
|--------|---------------------|---------------------------|-----------|
| **A. Teacher-continuation logits** (recommended) | Full distribution over next K=1..8 tokens from teacher | Qwen3 embedding of teacher's argmax continuation, or a learned projection of teacher's top-k logits | Clean signal. Requires teacher and student share tokenizer (✅ if teacher is Qwen3-72B or Qwen3-Max; ❌ if Claude). |
| **B. Teacher text advice** | Short natural-language hint ("use substitution not elimination") | Qwen3-embed the advice string; mean-pool → value vector | Works with Claude/GPT-4. Advice must be parseable + short. Loses logit-level granularity. |
| **C. Teacher-rewritten continuation** | Teacher writes the next full reasoning step | Student is trained to match teacher's next-step logits | Teacher-forcing style, large-scale SFT-like. Memory captures the "nudge" from student's natural continuation to teacher's. |
| **D. Teacher binary rating** | Teacher says "this step is correct/incorrect" + rationale | Treat as PRM signal; memory trained via weighted-CE | Equivalent to building a PRM into memory. Same-ish as Math-Shepherd + memory. |

**Decision recommendation**: Start with **Option A** using **Qwen3-72B-Thinking** (or Qwen3-Next-80B) as teacher — same tokenizer, logit-level signal, cheap inference on shared infra if available. If 72B not accessible, fall back to **Option B with Claude** as a text-advice teacher. Make Option A vs B an ablation: does logit-level vs text-level advice differ materially?

### C2. How is entropy computed and thresholded?

- Compute **predictive entropy** over the Qwen3 vocabulary at each generated position during student rollout.
- Exclude low-info positions: whitespace, LaTeX boilerplate, closing `$`, `}` when nested brace count > 0.
- Threshold: **top-20% by entropy** within each generation (Wang 2025 rule). Calibrate per dataset — MATH vs algebra will differ.
- Gate additionally by **step boundary**: prefer entropy-high tokens that are also at newline / after `=` / after "therefore" / at start of step. This layered gate yields ~2% of tokens. Prevents capturing intra-word entropy spikes which aren't semantic branches.

### C3. Which hidden-state to use as memory KEY?

- `h_t` from layer **L ∈ {14, 18, 22}** of Qwen3-8B (mid-depth; research files 01, 02, 07 converge on this range).
- Use the hidden state **just before the branching token is emitted** — causal masking ensures it doesn't see the answer.
- Consider concatenating the hidden state from multiple layers (e.g. [L14, L18, L22]) to capture hierarchical features; adds parameters but richer keys.

### C4. Memory architecture — param budget

At Qwen3-8B's 4096 hidden dim, naive port of our 270K-param memory explodes. Use a **bottleneck**:

```python
class BottleneckKVMemory(nn.Module):
    def __init__(self, n_slots=256, bottleneck=256, hidden=4096):
        self.down = nn.Linear(hidden, bottleneck, bias=False)      # 1.05M
        self.K = nn.Parameter(torch.randn(n_slots, bottleneck))    # 65K
        self.V = nn.Parameter(torch.zeros(n_slots, bottleneck))    # 65K (init zero!)
        self.up = nn.Linear(bottleneck, hidden, bias=False)        # 1.05M
        self.gate = nn.Parameter(torch.zeros(1))                   # 1
        # Total: ~2.2M trainable params
    def forward(self, h):
        q = self.down(h)                                            # B,T,256
        a = F.softmax(q @ self.K.T / (bottleneck ** 0.5), dim=-1)   # B,T,n_slots
        v = a @ self.V                                              # B,T,256
        return self.gate * self.up(v)                               # B,T,4096
```

Initial output ≈ 0 because `V=0` and `gate=0`. Safe to hook into frozen backbone. 2.2M params is Phase-1 target; sweep n_slots ∈ {128, 256, 512, 1024} and bottleneck ∈ {128, 256, 512}.

Compare to baselines:
- LoRA r=32 all-linear on Qwen3-8B: ~41M trainable params (≈20× more than memory)
- DoRA r=32: similar
- TinyLoRA 13-param (2026): ridiculous lower bar

### C5. Loss function

At selected positions t_i (entropy-gated), with teacher's next-token distribution $p^T$:

$$L_{memory} = \sum_{i} \text{KL}(p^S(\cdot|h_{t_i} + \text{mem}(h_{t_i})) \| p^T(\cdot|s_{t_i}))$$

At non-selected positions, no loss (gradient masked). This matches the "top-20% entropy RLVR" rule from Wang 2025.

Optionally add a **capability-preservation regularizer** at random non-branching positions:

$$L_{preserve} = \text{KL}(p^S_{with\_memory}(\cdot) \| p^S_{frozen}(\cdot))$$

Weight this small (e.g. 0.1); it pulls the memory output toward zero outside its firing zone.

### C6. Training data scale

Per problem:
- ~1000 CoT tokens
- ~2% of tokens selected (entropy + step-boundary gate) → ~20 branching points per problem
- ~1000 training problems → ~20K teacher queries

Teacher cost:
- **Qwen3-72B local**: free on H100 cluster if available; otherwise cloud ~$0.50/1M tokens → $10
- **Claude Sonnet**: ~$3/1M in + $15/1M out. 20K queries × ~500 tokens average ≈ 10M tokens → ~$100
- **GPT-4o**: similar

All feasible for a research-scale run.

## Phase-by-phase execution

### Phase 1a — DP detection pipeline
1. Run Qwen3-8B zero-shot on MATH train set (1000 problems) with full logits logged.
2. Compute per-token entropy.
3. Implement step-boundary detector (regex + token-ID match for `\n`, `=`, common step-start words).
4. Visualize entropy + step-boundary overlap — reuse `visualize_superposition.py` UI structure. Sanity-check: high-entropy tokens concentrate at visible decision points (operation choice, substitution/elimination, cases split).
5. Output: `outputs/phase1a/dp_positions.jsonl` — for each problem, list of branching-token indices.

### Phase 1b — Teacher query pipeline
1. Pick teacher (decision C1).
2. At each DP index, build a partial-context prompt: problem + CoT-so-far.
3. Query teacher, collect advice (logits if same tokenizer, text if not).
4. Cache aggressively — teacher calls are the bottleneck. One cache key = (problem_hash, dp_position).
5. Output: `outputs/phase1b/teacher_advice.jsonl` with per-DP teacher response.

### Phase 1c — Memory training
1. Implement `BottleneckKVMemory` (C4).
2. Implement hook on Qwen3 at layer L (see `research/07` — Tensor-not-tuple gotcha).
3. Training loop: forward with hook → compute KL at DP positions → backprop → only memory params get grad.
4. Sweep: L ∈ {14, 18, 22}, n_slots ∈ {128, 256, 512}, bottleneck ∈ {128, 256}.
5. Output: trained memory checkpoints + standard JSON result schema (mirror `../naive_ft/ft.py:266-282`).

### Phase 1d — Evaluation
1. Inference WITHOUT teacher: Qwen3-8B + memory hook on MATH-500 test + algebra test.
2. Metrics: greedy + maj@16 with Wilson CI, per-MATH-level stratified (per `research/08`).
3. Ablations to run:
   - **No-memory baseline** (frozen Qwen3-8B only).
   - **Random position teacher** (same teacher cost, non-entropy-gated positions) — does entropy matter?
   - **Text-advice vs logit-advice** (C1 A vs B).
   - **LoRA distillation** (same teacher, same positions, LoRA r=32 all-linear) — does memory beat weights?
   - **Full-sequence distillation** (EOPD-style) — does entropy gating matter?
   - **Training-free CAMELoT control** (research/13) — does the learning earn its keep?
4. Capability-preservation check: run on MMLU, arithmetic, a Czech QA set. Expect memory-hooked model to match frozen baseline within ±1 pt. LoRA variant should show forgetting; that's the memory's selling point.

## Open questions (to resolve before coding Phase 1)

1. **Teacher choice.** Qwen3-72B-Thinking (cleanest), Claude (smartest but text-only), or ensemble?
2. **Advice shape.** Logits, text, or rewrites? (C1). Pick one for Phase 1, ablate others if it works.
3. **Memory key depth.** Single-layer hidden state or multi-layer concat?
4. **DP threshold.** Top-20% is a default — do we calibrate per dataset or fix globally?
5. **Training schedule.** Fixed teacher advice as targets (offline) or re-query each epoch (online, like EOPD)? Offline is much cheaper.
6. **When does memory fire at inference?** Every token or only when hidden-state similarity crosses a threshold? Early experiments can check if a gated variant helps.
7. **Preservation regularizer.** Needed or does the frozen backbone already preserve? Test both.

## Concrete "first working version" scope (minimum viable)

Cut all ablations, pick middle-of-the-road choices:
- Teacher: **Qwen3-72B-Thinking** via vLLM.
- Advice: **top-1 continuation from teacher for 4 tokens** (Option A, short horizon).
- Entropy gate: **top-20% × step-boundary**, ~2% of tokens.
- Memory: **n_slots=256, bottleneck=256, layer L=18**. ~2.2M params.
- Loss: **KL at DP positions, no preservation regularizer yet**.
- Training: 1000 MATH problems, 3 epochs, Adam 1e-3, bs=4, grad clip 1.0.
- Eval: MATH-500 test greedy accuracy, compared vs frozen baseline and vs matched-param LoRA distillation.

One week of engineering for the loop, one week for teacher queries + training, one week for ablations.

## Open literature actions

Before coding, verify novelty more rigorously by:
1. Reading EOPD (arXiv 2603.07079) and TIP (2604.14084) in full — these are the closest prior art.
2. Searching ACL/NeurIPS/ICLR 2026 for "memory module distillation" and "associative memory distillation".
3. Checking Engram paper's discussion section for related work that distills into memory.

If a paper matching our exact setup exists, pivot to either (a) a cleaner ablation of one factor (e.g. entropy-gate importance at matched teacher budget) or (b) a different distillation target (e.g. multi-teacher ensemble, or self-distillation with stronger prompting).

## Sources

Close prior art (must read):
- [EOPD — Entropy-Aware On-Policy Distillation (2603.07079, 2026)](https://arxiv.org/html/2603.07079)
- [TIP — Token Importance in On-Policy Distillation (2604.14084, 2026)](https://huggingface.co/papers/2604.14084)
- [Beyond 80/20 — High-Entropy Minority Tokens RLVR (Wang 2506.01939, 2025)](https://arxiv.org/abs/2506.01939)
- [Selective Knowledge Distillation SE-KD (2602.01395, 2026)](https://arxiv.org/html/2602.01395)
- [Thinking Machines — On-Policy Distillation blog (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Engram (DeepSeek, Jan 2026) — see `research/13` and `memory_papers/2601.07372v1.pdf`
- LongMem (2306.07174) — see `memory_papers/2306.07174v1.pdf`
- PKM-ResM (2010.03881) — see `memory_papers/2010.03881v1.pdf`

Internal references:
- `research/04_branching_decision_points.md` — entropy-based DP candidates
- `research/05_process_reward_models.md` — PRM alternative to teacher
- `research/10_2026_prm_and_critical_tokens.md` — 2026 step-level RL and critical-token work
- `research/13_local_memory_papers_review.md` — memory-module prior art
- `../visualization/visualize_superposition.py` — entropy-plot UI template for Phase 1a
- `../scripts/inference_decision_chains_extended.py:287-346` — DP position extraction template
