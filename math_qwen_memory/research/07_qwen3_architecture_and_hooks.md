# Qwen3-8B — Architecture, Quirks, and Hooking Guide

Engineering reference for porting our "FROZEN backbone + trainable residual-stream memory module" pattern from a 512-d LitGPT GPT-NeoX block to Qwen3-8B loaded via HuggingFace transformers. Summarises what we must know before we write the hook: exact block structure (it is *not* the same as LitGPT), return-type of the decoder layer (not a tuple!), param budget, GQA shape, thinking-mode mechanics, and KV-cache interaction during `generate()`.

## Model variants to consider

| ID | Purpose | When to use | Notes |
|---|---|---|---|
| `Qwen/Qwen3-8B` | Hybrid instruct with thinking toggle | **Default**. This is what we should target. | 32,768 native ctx, up to 131K with YaRN. `enable_thinking` switches mode per call. |
| `Qwen/Qwen3-8B-Thinking-2507` | Dedicated thinking-only | Upgraded reasoner; always emits `<think>…</think>` | 256K native ctx; greedy decoding forbidden (endless repetition). T=0.6, top_p=0.95. |
| `Qwen/Qwen3-8B-Instruct-2507` | Dedicated non-thinking | Natural control / speed baseline. Never emits `<think>` blocks. | 256K native ctx. T=0.7, top_p=0.8. |

Recommendation: build against `Qwen/Qwen3-8B` first (single hybrid checkpoint, enable_thinking toggle), then optionally validate against the `-2507` pair as ablation endpoints.

## Architecture specs (`config.json`, Qwen3-8B)

Pulled verbatim from HF `config.json`:

| Field | Value |
|---|---|
| `model_type` | `qwen3` |
| `architectures` | `["Qwen3ForCausalLM"]` |
| `num_hidden_layers` | **36** |
| `hidden_size` | **4096** |
| `intermediate_size` | 12288 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | **8** (GQA, 4:1) |
| `head_dim` | 128 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 40960 (32K native + 8K buffer) |
| `rope_theta` | 1,000,000 |
| `rope_scaling` | null (YaRN must be enabled manually for long ctx) |
| `rms_norm_eps` | 1e-6 |
| `hidden_act` | silu |
| `attention_bias` | false |
| `sliding_window` | null, `use_sliding_window`: false |
| `tie_word_embeddings` | **false** (untied, unlike Qwen3-0.6B / 1.7B) |
| `torch_dtype` | bfloat16 |
| `bos_token_id` | 151643 (`<|endoftext|>`) |
| `eos_token_id` | 151645 (`<|im_end|>`) |

Compared with **Qwen2.5-7B**: layers 28 → **36**, hidden 3584 → **4096**, heads 28 → 32, kv-heads 4 → **8**, intermediate 18944 → 12288 (narrower MLP, deeper stack), rope_theta 1M (same), context 128K → 32K native (but YaRN to 131K). Qwen2.5-7B tied embeddings; Qwen3-8B does **not** tie. Vocab grew by ~3k (new thinking + vision placeholder tokens).

## Block diagram of `Qwen3DecoderLayer`

From `transformers.models.qwen3.modeling_qwen3` (v4.56), layer forward is pre-norm with two additive sub-blocks:

```
x_in ──────────────────────────────────┐ residual
  │                                     │
  ├─► input_layernorm (RMSNorm) ─► self_attn ─► + ◄─ residual
  │                                     ▼
  │                                    x1 ────────────────┐ residual
  │                                     │                  │
  │                                     ├─► post_attention_layernorm ─► mlp ─► + ◄─
  ▼                                                                            │
                                                                              x_out
```

Concretely: `Qwen3Attention` uses Q/K **head-dim RMSNorm** (`q_norm`, `k_norm`) applied after the projections and before RoPE — this is a Qwen3-specific quirk (Qwen2 did not have it). `Qwen3MLP` is the standard SwiGLU (`gate_proj`, `up_proj`, `down_proj`). Final `Qwen3Model` applies one more RMSNorm (`self.norm`) after the last layer before `lm_head`.

## Where to hook for "residual stream after layer L"

Object path on a loaded `Qwen3ForCausalLM` model:

- `model.model.embed_tokens` — token embedding (trainable-frozen)
- `model.model.layers[L]` — a `Qwen3DecoderLayer` (L in [0, 35])
- `model.model.norm` — final RMSNorm
- `model.lm_head` — untied output projection

Hook `model.model.layers[L]` to read/write the residual stream **after** layer L (i.e. after both sub-blocks plus their residual adds).

### Critical gotcha: layer forward returns a Tensor, NOT a tuple

In older HF models (Llama ≤ some versions, Qwen2), `decoder_layer(...)` returned `(hidden_states, ...)` and hooks had to unpack a tuple. In current transformers (≥4.52ish, definitely 4.56), `Qwen3DecoderLayer.forward` is typed `-> torch.Tensor` and returns the tensor directly:

```python
# transformers/models/qwen3/modeling_qwen3.py
def forward(self, ...) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(...)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states      # <── bare Tensor, not a tuple
```

So our hook must also return a bare Tensor (not `(tensor,)`), otherwise the next layer receives a tuple and crashes with a shape error. The correct pattern:

```python
def hook(module, args, output):
    # output is torch.Tensor of shape [B, T, H]
    hidden = output
    delta = memory(hidden)          # trainable module, same dtype/device
    return hidden + delta           # bare Tensor, SAME dtype
handle = model.model.layers[L].register_forward_hook(hook)
```

If you need tuple-compatibility across transformers versions (defensive), detect with `isinstance(output, tuple)`:

```python
def hook(module, args, output):
    if isinstance(output, tuple):
        hidden = output[0]
        new = hidden + memory(hidden)
        return (new,) + output[1:]
    else:
        return output + memory(output)
```

Always cast the memory-module output to `hidden.dtype` (bf16) before addition — otherwise a fp32 delta silently upcasts the residual stream and flash-attn downstream will error out.

## Thinking mode mechanics

The base `Qwen3-8B` is a **hybrid**: the chat template decides whether a `<think>…</think>` reasoning segment is requested.

- `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)` — appends nothing special after `<|im_start|>assistant\n`; the model is expected to emit `<think>` itself and close it before producing the answer.
- `enable_thinking=False` — template appends a literal `<think>\n\n</think>\n\n` pre-filled block so the model is forced into direct-answer mode.
- **Soft switch**: user can embed `/think` or `/no_think` tokens in their last message; the template respects per-turn overrides.
- Special token IDs: `<think>` = **151667**, `</think>` = **151668**. These are in the tokenizer but not referenced in `config.json`; find them by `tokenizer.convert_tokens_to_ids("<think>")`.
- Parsing at eval time: find last occurrence of token id 151668 to split `thinking_content` from final `content`.

For training with our memory module: decide whether to train WITH thinking (hook fires during CoT generation → potentially useful signal, longer sequences, more compute) or WITHOUT (fix `enable_thinking=False`, simpler, more like the old decision-chain setup). I'd start with `enable_thinking=False` and add thinking as a second experiment — it isolates the memory-module contribution from the CoT contribution.

## Special tokens & chat template

| Token | ID | Role |
|---|---|---|
| `<|endoftext|>` | 151643 | `bos_token_id`, pad_token, generic doc boundary |
| `<|im_start|>` | 151644 | role marker open |
| `<|im_end|>` | 151645 | `eos_token_id`, role marker close |
| `<think>` / `</think>` | 151667 / 151668 | CoT delimiters |
| vision/object/box pads | 151646–151656 | unused for text-only Qwen3 |

`bos_token` is `null` in `tokenizer_config.json` — do **not** prepend a BOS; the chat template already emits `<|im_start|>system\n…<|im_end|>`. Pad is `<|endoftext|>` (151643). EOS is `<|im_end|>`, **not** `<|endoftext|>` — set generation `eos_token_id=151645` (or list both for safety).

Qwen3 chat template is a superset of Qwen2's: identical `<|im_start|>role\n…<|im_end|>\n` framing, plus conditional `<think>…</think>` blocks and the `/think`, `/no_think` soft-switch logic baked into the Jinja template. Tool-calling format is enriched relative to Qwen2.

## Attention implementation notes

`Qwen3PreTrainedModel` advertises:

- `_supports_sdpa = True`
- `_supports_flash_attn = True` (i.e. flash-attn 2 via `FlashAttentionKwargs` plumbing)
- `_supports_flex_attn = True`

Load with `attn_implementation="flash_attention_2"` when flash-attn is installed and your GPU supports it (Ampere+ / Hopper). Otherwise `sdpa` (PyTorch ≥2.1) is the default and is fine for our purposes. **Important**: attention forward dispatches through `ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]` at every call, meaning a runtime-changed `_attn_implementation` is picked up live — useful for debugging, don't toggle during training.

Our residual-stream hook is **downstream** of the attention internals, so the attention backend is irrelevant to hook correctness. GQA (32 Q / 8 KV heads, 4:1) is also irrelevant for residual-stream hooks — the `[B, T, H]` tensor at the layer boundary has no GQA exposure.

## KV cache interaction at generation

During `model.generate(..., use_cache=True)`:

1. On step 0 (prefill), the hook sees `hidden_states` of shape `[B, T_prompt, 4096]` — full prompt tokens.
2. On each subsequent decode step, the hook sees `[B, 1, 4096]` — only the newly-generated token's residual at layer L.

Implications:

- The memory module's forward must be fast enough to run at every decode step (36 layers × N generated tokens × B batch, per decode forward). For Qwen3-8B a single decode step at bf16 on an H100 is ~15-25ms total; a cross-attention or deep MLP memory module added to one layer is fine, added to all 36 layers would dominate.
- Do **not** stash state on `self` in the memory module keyed by `[B, T]` — T changes between prefill and decode, and B can change across calls.
- KV cache is held by the attention sub-module (`past_key_values`). Your residual-stream hook lives *after* self_attn has already updated the cache, so modifying `output` does not re-enter the cache. This is what we want: we perturb the residual that flows into layer L+1, but layer L's KV entries reflect the un-perturbed attention pass.
- If you later want to hook *before* self_attn (so modifications also propagate into KV), hook `model.model.layers[L].input_layernorm` output or pre-hook `model.model.layers[L]`, but be careful of `position_embeddings` that are passed as a sibling kwarg.

## Param count sanity check

At `H=4096, I=12288, heads=32, kv_heads=8, head_dim=128, L=36, V=151936`:

| Block | Params | Notes |
|---|---|---|
| Per layer attention (q/k/v/o, no bias) | ~**42.0 M** | `Q: 16.78M, K: 4.19M, V: 4.19M, O: 16.78M` |
| Per layer MLP (SwiGLU) | ~**151.0 M** | `3 × H × I` |
| Per layer norms (input_ln + post_attn_ln + q_norm + k_norm) | ~0.009 M | negligible |
| **Per layer total** | ~**193 M** | |
| 36 layers | ~6.95 B | matches "non-embedding 6.95 B" in model card |
| Token embed | 622 M | `V × H` |
| LM head (untied) | 622 M | `V × H` |
| Final RMSNorm | 4k | negligible |
| **Grand total** | ~**8.19 B** | matches advertised 8.2 B |

Memory module budget at `H=4096`:

- A single 1-layer transformer block (self-attn + MLP) at H=4096 ≈ **193 M**. Inserting this at one layer adds ~2.8% params — cheap.
- A simple 2-layer MLP (`H → 4H → H`) with biases ≈ `2 × 4 × H^2` = **134 M**. Also cheap at one site.
- A gated memory lookup (K keys × H) with K=1024 ≈ **4 M** — trivial.
- **Avoid**: inserting at *every* layer with a 193M module → +6.95 B params doubles the model.

Rule of thumb for our 4096-d residual: keep per-site memory-module params under ~200 M, insert at 1–4 sites, keep total added params under ~800 M (~10% of frozen backbone).

## Code sketch: minimal memory-module wrapper for Qwen3

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ResidualMemory(nn.Module):
    """Trainable perturbation applied to layer-L residual stream."""
    def __init__(self, hidden_size: int, bottleneck: int = 1024):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        nn.init.zeros_(self.up.weight)        # start as identity perturbation
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(h)))

class Qwen3MemoryHook:
    def __init__(self, model, layer_idx: int, memory: nn.Module):
        self.memory = memory
        layer = model.model.layers[layer_idx]
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        # Qwen3DecoderLayer returns a bare Tensor [B, T, H]
        if isinstance(output, tuple):          # defensive across HF versions
            hidden = output[0]
            delta = self.memory(hidden).to(hidden.dtype)
            return (hidden + delta,) + output[1:]
        hidden = output
        delta = self.memory(hidden).to(hidden.dtype)
        return hidden + delta

    def remove(self):
        self.handle.remove()


# Wiring
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",   # or "sdpa"
    device_map="cuda",
)
for p in model.parameters():
    p.requires_grad_(False)                    # freeze backbone

memory = ResidualMemory(hidden_size=4096, bottleneck=1024).to(
    dtype=torch.bfloat16, device="cuda",
)
hook = Qwen3MemoryHook(model, layer_idx=18, memory=memory)     # mid-stack

# Train only `memory.parameters()`; backbone frozen.
opt = torch.optim.AdamW(memory.parameters(), lr=1e-4)

# Generation works transparently (hook fires on prefill + every decode step):
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "2+2=?"}],
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
).to("cuda")
out = model.generate(prompt, max_new_tokens=64, eos_token_id=151645,
                     do_sample=True, temperature=0.7, top_p=0.8)
```

Three things to double-check the first time you run this:

1. `memory.parameters()` are the only ones with `requires_grad=True` — verify with `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
2. Hook fires on both training (`model(input_ids, labels=...)`) and `model.generate(...)` without additional plumbing.
3. When you do `model.save_pretrained(...)`, the memory module is NOT saved (it's not part of `model`). Save it separately: `torch.save(memory.state_dict(), "memory.pt")`.

## Sources

- Qwen3-8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3-8B `config.json`: https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- Qwen3-8B `tokenizer_config.json`: https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json
- Local modeling file (transformers 4.56.2): `transformers/models/qwen3/modeling_qwen3.py` — `Qwen3DecoderLayer.forward`, `Qwen3Attention`, `Qwen3Model.forward`.
- Qwen3-2507 variants guide (Unsloth): https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-2507
- Qwen3-8B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-8B-Instruct-2507
- Qwen3-8B-Thinking-2507: https://huggingface.co/Qwen/Qwen3-8B-Thinking-2507
- Fireworks Qwen3 model-selection guide: https://fireworks.ai/blog/qwen-3-decoded
