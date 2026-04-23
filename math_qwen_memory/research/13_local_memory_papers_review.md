# Local Memory Papers Corpus — Triage & Relevance Review

Source: `/mnt/raid/data/Hyner_Petr/operations_entropy/DecisionChains/memory_papers/` (10 PDFs, downloaded 2026-04-15).

Each paper scored **H / M / L** for relevance to our port of a frozen-backbone + trainable external-memory module to Qwen3-8B on natural-language math reasoning. H = direct methodological match or strong empirical evidence on our target; M = adjacent; L = background only.

---

## H. Directly relevant — read first

### [H] 2010.03881 — *Large Product Key Memory for Pretrained Language Models* (Kim & Jung, Clova AI, Oct 2020)
- PKM (Lample 2019) applied to **pretrained** LMs (BERT). Two key tricks:
  1. **Init from PLM weights trained WITHOUT memory**, then add memory; avoids catastrophic drift.
  2. **Residual memory (ResM)**: ADD PKM output to FFN output rather than REPLACING FFN. Keeps arithmetic intact.
- Observation: only few memory slots get used → "catastrophic drift" from sparse gradient. Both init + residual connection help.
- **Why it matters for us**: This is the closest methodological ancestor to our memory module. We're doing the same thing — frozen pretrained backbone + trainable memory injected at residual stream. The ResM "add not replace" pattern is exactly what our hook does. Lessons transfer directly.
- **Concrete takeaways**:
  - Initialize memory keys/values so initial residual add ≈ 0 (don't shock the frozen model).
  - Monitor memory-slot utilization during training; sparse activation is a known failure mode.
  - Residual add is correct; do not try to replace an MLP block.
- **Updates our recommendations in**: file 01 (add this as canonical citation for the "add vs replace" design choice).

### [H] 2306.07174 — *LongMem: Augmenting LMs with Long-Term Memory* (Wang et al, MSR/UCSB, Jun 2023)
- **Frozen backbone LLM + trainable "Residual SideNet"** acting as memory retriever/reader. Cached KV pairs from earlier segments written to a non-differentiable memory bank. SideNet cross-attends to that bank, with cross-network residual connections back into the frozen LLM's hidden states.
- 65K-token memory; designed for long-context + many-shot in-context learning. Avoids memory staleness (the MemTRM problem) via decoupling.
- **Why it matters for us**: Architecturally this is the *exact template* for our Phase-1 design — frozen backbone + small trainable side-module that writes back into the residual stream. Only difference is what goes into the bank (LongMem: past KV; us: learnable K,V parameters).
- **Concrete takeaways**:
  - **Use the SideNet pattern**: a small trainable network that reads hidden state, consults memory, writes back via residual connection. Our memory module fits this mold.
  - **Avoid memory staleness**: if the bank contains cached representations from the frozen model, those stay valid. Learnable K,V don't drift either. Safe design.
  - **Cross-attn fusion** is their fusion mechanism — compare to our simple additive hook.
- **Updates our recommendations in**: files 01, 09.

### [H] 2510.15103 — *Continual Learning via Sparse Memory Finetuning* (Lin/Zettlemoyer et al, FAIR Meta, Oct 2025)
- **Memory-layer finetuning** where only top-t most-activated slots are updated per batch (TF-IDF-ranked vs pretraining activations). Builds on Meta's Memory Layers (Berges 2024) and PEER (He 2024).
- **Headline numbers** on continual-learning QA: NaturalQuestions F1 drops **89% with full FT, 71% with LoRA, only 11% with sparse memory FT** — all at matched new-knowledge acquisition.
- **Why it matters for us**: Best current empirical evidence that sparse updates to a memory-layer beat LoRA at preserving prior capability. Exactly the "add capability without forgetting" trade-off our frozen-backbone approach targets. Meta's Memory Layers are the standardized primitive.
- **Concrete takeaways**:
  - Adopt **sparse per-batch slot updates** as an ablation against dense-memory training.
  - If we train a memory bank: track per-slot utilization; update only activated slots.
  - Forgetting metric: evaluate baseline-Qwen3-8B capabilities (MMLU, arithmetic) pre- and post-training to quantify interference.
- **Updates**: file 09 (new 2025 data point superseding older continual-FT claims). Also file 06 (LoRA-forgetting-vs-sparse-memory-FT contrast).

### [H] 2601.00671 — *Fast-weight Product Key Memory (FwPKM)* (Zhao & Jones, Sakana AI, **Feb 2026**)
- PKM keys/values updated via **Test-Time Training**-style gradient updates on a sparse memory at inference. Chunk-level rewrite loss. Sparse activation (~small fraction slots per token).
- Needle-in-Haystack: 4K-trained → **128K generalization**, retrieval accuracy <10% → >70% with iterative reading.
- Code: https://github.com/SakanaAI/fast-weight-product-key-memory
- **Why it matters for us**: The **most recent 2026 evolution of PKM**, the literal family our memory module belongs to. Shows inference-time gradient writes to memory can work. Possible extension: have our memory bank update at inference (per-problem episodic memory) instead of only learning at training time.
- **Concrete takeaways**:
  - Read their implementation for modern PKM idioms (sparse activation, chunking, init).
  - Consider a TTT variant as Phase-3 ablation: per-problem memory updates during CoT generation.
  - Use their interpretability analyses (Sec 5) as template for analyzing which slots our memory activates.
- **Updates**: file 01 (add as the canonical modern PKM reference) and file 09 (2026 milestone missed by that agent).

### [H] 2601.07372 — *Engram: Conditional Memory via Scalable Lookup* (Cheng et al, DeepSeek-AI / Peking, **Jan 2026**)
- Modernized N-gram embedding as **conditional memory sparsity axis**, complementary to MoE. 27B model with Engram.
- **Results directly on our benchmarks**: MMLU **+3.4**, BBH **+5.0**, ARC-Challenge **+3.7**, HumanEval **+3.0**, **MATH +2.4**, **GSM8K +2.2**.
- Mechanistic finding: Engram relieves backbone early layers from static reconstruction → deepens effective network for reasoning.
- Code: https://github.com/deepseek-ai/Engram
- **Why it matters for us**: **Direct empirical evidence that memory-style modules help math reasoning** on the exact benchmarks we're targeting. A key source of support for the premise that the method family is worth porting.
- **Concrete takeaways**:
  - MATH +2.4, GSM8K +2.2 is our order-of-magnitude expectation if memory-steering works at 8B scale.
  - Their "U-shaped sparsity allocation" law informs how we split capacity between backbone and memory.
  - Deterministic addressing enables CPU prefetch — architectural tip if we scale beyond 40 GB.
- **Updates**: file 09 (missed this paper!) and file 01.

---

## M. Adjacent — skim for specific ideas

### [M] 2402.13449 — *CAMELoT: Training-Free Consolidated Associative Memory* (He/Krotov et al, UCSD/IBM, Feb 2024)
- **Training-free** plug-and-play associative memory (Krotov-style) on any frozen attention LLM. Novelty/recency-managed slot updates. Long-context PPL −29.7% vs LLaMA.
- **Why it matters**: Shows a frozen-model memory module with **zero gradient training** can work. That's a strong baseline to beat; if CAMELoT-style works without training, our trainable version must demonstrate it earns its gradients.
- **Takeaway**: Include training-free associative-memory as an ablation arm. If our trained memory doesn't beat a CAMELoT-style control, the learning isn't doing work.

### [M] 2407.01437 — *Needle in the Haystack for Memory Based LLMs* (Nelson et al, IBM, Jul 2024)
- Tests **Larimar** (external associative memory with least-squares updates) on 100K–1M token recall. 1.3B model, memory stored off-GPU (CPU).
- **Why it matters**: Demonstrates feasibility of CPU-offloaded memory, useful if we later want a memory bank bigger than 40 GB allows on-GPU.
- **Takeaway**: Probably not Phase-1 (we're not doing long-context), but file away for scale-ups. Larimar's least-squares read/write is an alternative to our softmax-attention memory.

### [M] 2407.01178 — *Memory³* (Yang et al, IAAR Shanghai / PKU, Jul 2024)
- 2.4B LLM pretrained from scratch with **explicit sparse KV memory**. Defines a memory hierarchy (model params / explicit memory / RAG context). Outperforms larger LLMs on benchmarks.
- **Why it matters**: Conceptual framing — memory as a third store alongside weights and context. Useful lens but not a drop-in method since they pretrain from scratch (not compatible with 40 GB A100).
- **Takeaway**: Use their cost equation (Eq 1 in the paper) to frame when external memory is worth it over more params. Skip the architecture; adopt the framing.

### [M] 2601.21461 — *L³: Large Lookup Layers* (Tseng & De Sa, Cornell, **Jan 2026**)
- Context-dependent **static token-keyed embedding lookup**; generalizes tokenizer embedding table to decoder layers. Outperforms iso-FLOP dense and iso-sparse MoE at 0.8–2.6B.
- **Why it matters**: Alternative conditional-memory primitive keyed on **current token id** rather than hidden state. Orthogonal to MoE. Systems-friendly (CPU offload).
- **Takeaway**: Candidate Phase-3 ablation — what if we key memory by input-token id instead of hidden-state query? Same parameter budget, much simpler, might reveal which axis of conditioning matters.

---

## L. Background only — no direct action

### [L] 2506.01963 — *Breaking Quadratic Barriers* (Kiruluta et al, Berkeley, Jun 2025)
- Non-attention architecture: SSM + multi-res conv + retrieval-augmented external memory. Benchmarked on WikiText-103 / Enwik8.
- **Why skip**: It's an alternative LM architecture, not a frozen-model intervention. Incompatible with "take Qwen3, add a module" framing. Interesting for ultra-long context but outside our scope.

---

## Cross-reference updates

Papers in this corpus that our existing 01/09 reviews **missed or under-cited**:

| Our file | Missing citation |
|---------|-----------------|
| 01 | **2010.03881 (Kim 2020, PKM+ResM)** — should be headline citation for "add not replace" design principle |
| 01 | **2306.07174 (LongMem)** — should be headline citation for "frozen + residual SideNet" pattern |
| 09 (2026) | **2601.00671 (FwPKM, Feb 2026)** — belongs in the 2026 memory section |
| 09 (2026) | **2601.07372 (Engram, Jan 2026)** — MATH +2.4 / GSM8K +2.2 is the most directly relevant 2026 result |
| 09 (2026) | **2601.21461 (L³, Jan 2026)** — orthogonal conditional-memory axis |
| 06 (LoRA) | **2510.15103 (Sparse Memory FT, Oct 2025)** — sparse memory FT loses only 11% vs LoRA's 71% on continual-learning forgetting; strong argument for memory over LoRA in continual scenarios |

## Concrete Phase-1 design implications

1. **Adopt the LongMem SideNet pattern** as our Phase-1 skeleton (file 01 recommendation confirmed).
2. **Ensure residual add, not replace** (2010.03881 Kim 2020 rule).
3. **Init memory so initial output ≈ 0** to avoid shocking the frozen model (2010.03881).
4. **Monitor per-slot activation** during training; use sparse-update variant as ablation (2510.15103 + 2010.03881).
5. **Study FwPKM code** (2601.00671) for modern PKM init/sparsity tricks before writing our own.
6. **Baseline to beat**: match or exceed Engram's +2.4 MATH / +2.2 GSM8K gains (2601.07372). If our 8B result is smaller, the method likely doesn't scale below 27B.
7. **Include training-free associative memory as a control arm** (2402.13449 CAMELoT) — if it works without training, we must show ours adds value.
8. **Phase-3 ideas** to file away: L³-style token-id-keyed lookup, FwPKM-style TTT per-problem writes, Larimar-style CPU-offload if we scale the bank.

## Reading order (if constrained)

1. **2010.03881** (PKM+ResM, 2020) — foundational; 30 min.
2. **2306.07174** (LongMem, 2023) — closest architecture match; 45 min.
3. **2601.07372** (Engram, Jan 2026) — best recent math evidence; 30 min.
4. **2601.00671** (FwPKM, Feb 2026) — modern PKM, has code to clone; 45 min.
5. **2510.15103** (Sparse Memory FT, 2025) — forgetting story; 30 min.
6. Optional: 2402.13449 (CAMELoT), 2601.21461 (L³).
7. Skip: 2506.01963 (non-attention arch), 2407.01437 (1M context), 2407.01178 (pretrain from scratch) unless specific questions arise.

## Sources (local files)

- `memory_papers/2010.03881v1.pdf` — Kim & Jung, *Large Product Key Memory for Pretrained LMs*
- `memory_papers/2306.07174v1.pdf` — Wang et al, *LongMem: Augmenting LMs with Long-Term Memory*
- `memory_papers/2402.13449v1.pdf` — He/Krotov et al, *CAMELoT*
- `memory_papers/2407.01178v1.pdf` — Yang et al, *Memory³*
- `memory_papers/2407.01437v2.pdf` — Nelson et al, *Needle-in-the-Haystack for Memory LLMs / Larimar*
- `memory_papers/2506.01963v1.pdf` — Kiruluta et al, *Breaking Quadratic Barriers*
- `memory_papers/2510.15103v1.pdf` — Lin/Zettlemoyer et al, *Continual Learning via Sparse Memory Finetuning*
- `memory_papers/2601.00671v2.pdf` — Zhao & Jones, *Fast-weight Product Key Memory (FwPKM)*
- `memory_papers/2601.07372v1.pdf` — Cheng et al (DeepSeek), *Engram: Conditional Memory via Scalable Lookup*
- `memory_papers/2601.21461v2.pdf` — Tseng & De Sa, *L³: Large Lookup Layers*
