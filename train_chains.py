"""
Training script for the Chain Experiment.

Chains 2 anonymized functions (letters a-j) with traces.
Tests compositional generalization on novel function pairs + novel vectors.
"""

import sys
import os
import json
import torch
import numpy as np
import wandb
import lightning as L
import hydra
import logging

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import get_cosine_schedule_with_warmup
from typing import Optional

from framework.data import get_data, get_tokenizer, Datamodule
from framework.hf_config import get_configs as _get_configs

# Shim: original code uses hf_config.get_configs(cfg)
import types
hf_config = types.SimpleNamespace(get_configs=_get_configs)
from litgpt import LLM
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.api import Preprocessor
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files
from transformers import AutoModelForCausalLM
from pathlib import Path

from evaluation.evaluator_chains_extended import ExtendedChainEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def convert_litgpt_to_hf(cfg):
    """Convert LitGPT checkpoint to HuggingFace format for generation."""
    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)

    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, out_dir / "pytorch_model.bin")
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    return hf_model


class LitChains(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, train_batches, delimiter_token_id):
        super().__init__()
        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.train_batches = train_batches
        self.delimiter_token_id = delimiter_token_id
        _, self.hf_conf = hf_config.get_configs(cfg)

    def setup(self, stage):
        self.hf_conf["bos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[BOS]")
        self.hf_conf["eos_token_id"] = self.preprocessor.tokenizer.convert_tokens_to_ids("[EOS]")
        self.hf_conf["vocab_size"] = len(self.preprocessor.tokenizer.get_vocab())

        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        delimiter_positions = (input_ids == self.delimiter_token_id)
        first_search_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        first_search_pos[:, 1:] = delimiter_positions.cumsum(dim=1)[:, :-1].bool()
        mask = ~first_search_pos.cumsum(dim=1).bool()
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

    def training_step(self, batch, batch_idx):
        idx = batch["input_ids"]
        targets = self.mask_targets(idx, batch["labels"])
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch["input_ids"]
        targets = self.mask_targets(idx, batch["labels"])
        out, loss = self(idx, targets)

        predictions = torch.argmax(out, dim=-1)
        valid_mask = (targets[:, 1:] != -100)
        correct = (predictions[:, :-1] == targets[:, 1:]) * valid_mask
        total_correct = correct.sum()
        total_tokens = valid_mask.sum()
        accuracy = total_correct.float() / total_tokens if total_tokens > 0 else torch.tensor(0.0)

        self.log("acc", accuracy, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        test_dataset = self.trainer.datamodule.dataset["test"]
        tokenizer = self.preprocessor.tokenizer

        # Save and convert checkpoint for HF generation
        save_path = self.cfg.convert_hf.in_path
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)
        self.llm.model.to(self.device)

        hf_model = convert_litgpt_to_hf(self.cfg)

        # Load raw test JSON for decision-function letter info (if available).
        # Apply the same sampling as get_data() so indices align with the
        # tokenized test dataset.
        raw_test_data = None
        try:
            from hydra.utils import to_absolute_path
            test_file = to_absolute_path(self.cfg.data.test_file)
        except Exception:
            test_file = self.cfg.data.test_file
        try:
            with open(test_file, "r") as f:
                raw = json.load(f)
            if raw and "letter1" in raw[0] and "letter2" in raw[0]:
                # Mirror get_data() sampling: test split uses sample_test_set,
                # but the evaluator uses the "test" split which is loaded from
                # the same file as "val". Both "val" and "test" splits come
                # from test_file. Apply the test split sampling if configured.
                if getattr(self.cfg.data.sampling, "sample_test_set", False):
                    n = int(self.cfg.data.sampling.num_test)
                    raw = raw[:n]
                raw_test_data = raw
        except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError):
            pass

        evaluator = ExtendedChainEvaluator(
            config=self.cfg,
            tokenizer=tokenizer,
            split_str=self.cfg.data.split_str,
            hf_model=hf_model,
            batch_size=self.cfg.eval.batch_size,
            num_examples=self.cfg.eval.num_examples,
            raw_test_data=raw_test_data,
        )

        metrics, example_rows = evaluator.evaluate(test_dataset)

        # Log metrics
        for key, value in metrics.items():
            self.log(f"Chain/{key}", value, on_epoch=True, sync_dist=True,
                     prog_bar=(key in ("exact_match", "avg_prob_f", "avg_prob_g", "valid_chain_frac", "valid_solution_frac")))

        # Log WandB examples table
        if wandb.run is not None and example_rows:
            columns = list(example_rows[0].keys())
            table = wandb.Table(columns=columns)
            for row in example_rows:
                table.add_data(*[row[c] for c in columns])
            wandb.log({"chain_examples": table}, step=self.global_step)

        print("\n=== Chain Metrics ===")
        for key, value in sorted(metrics.items()):
            if not key.startswith("func_"):
                print(f"  {key}: {value:.4f}")

    def configure_optimizers(self):
        if self.cfg.optim.lr_type in ("linear", "linear-reg"):
            optimizer = torch.optim.AdamW(
                self.llm.model.parameters(),
                lr=self.cfg.optim.lr,
                weight_decay=self.cfg.optim.weight_decay,
                betas=(0.9, 0.95),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: (step + 1) / self.cfg.optim.warmup_steps
            )
            return [optimizer], [scheduler]
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.lr)
            scheduler = {
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optim.warmup_steps,
                    num_training_steps=self.cfg.optim.n_steps,
                ),
                "interval": "epoch",
            }
            return [optimizer], [scheduler]

    def forward(self, idx, targets=None):
        return self.llm(idx, targets)


@hydra.main(config_path="config", config_name="base_decision_chains_extended", version_base=None)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print(f"Model: {cfg.model.n_layer}L-{cfg.model.n_head}H-{cfg.model.n_embd}D")

    tokenizer = get_tokenizer(cfg)
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    conf.padded_vocab_size = len(tokenizer.get_vocab())
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)

    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, cfg.model.batch_size, cfg.data.num_workers, tokenizer)
    data.connect(max_seq_length=cfg.model.block_size)
    data.setup()

    train_size = len(data.train_dataloader())
    trace_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=True)[0]

    lit_model = LitChains(
        cfg=cfg, model=model, train_batches=train_size,
        preprocessor=preprocessor, delimiter_token_id=trace_token_id,
    )

    logger = WandbLogger(
        project=cfg.wandb.proj_name, name=cfg.model.name, config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",
        dirpath=f"outputs/temp/checkpoints/{cfg.model.name}",
        filename="{epoch:02d}-{acc:.4f}",
        save_top_k=2,
        mode="max",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()
