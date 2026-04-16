from litgpt.config import configs, Config, name_to_config
from omegaconf import DictConfig, OmegaConf


def get_configs(cfg: DictConfig):
    conf = Config(
        **dict(
            name="pythia-14m",
            hf_config=dict(org="EleutherAI", name="pythia-14m"),
            block_size=cfg.model.block_size,
            n_layer=cfg.model.n_layer,
            n_embd=cfg.model.n_embd,
            n_head=cfg.model.n_head,
            intermediate_size=cfg.model.n_embd * 4,
            padding_multiple=128,
            padded_vocab_size=None, # cfg.model.vocab_size,
        )
    )

    hf_conf = {
        "architectures": ["GPTNeoXForCausalLM"],
        "bos_token_id": None, # cfg.model.bos_id,
        "classifier_dropout": 0.1,
        "eos_token_id": None, # cfg.model.eos_id,
        "hidden_act": "gelu",
        "hidden_size": cfg.model.n_embd,
        "initializer_range": 0.02,
        "intermediate_size": cfg.model.n_embd * 4,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": cfg.model.block_size,
        "model_type": "gpt_neox",
        "num_attention_heads": cfg.model.n_head,
        "num_hidden_layers": cfg.model.n_layer,
        "rotary_emb_base": 10000,
        "rotary_pct": 0.25,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.29.2",
        "use_cache": True,
        "use_parallel_residual": True,
        "vocab_size": None # cfg.model.vocab_size,
    }
    return conf, hf_conf