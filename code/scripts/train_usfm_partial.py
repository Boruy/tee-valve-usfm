"""Train official USFM SegVit with early ViT blocks frozen and separate learning rates.

Launched by run_usfm.py. The official Hydra configuration and trainer remain in
external/USFM; this project owns only the freezing and optimizer policy.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig


UPSTREAM = Path(__file__).resolve().parents[2] / "external/USFM"
sys.path.insert(0, str(UPSTREAM))

import usdsgen.trainer as trainer_package  # noqa: E402
from usdsgen.utils.lr_scheduler import build_scheduler  # noqa: E402


class _ProgressLoader:
    """Log completed batches while retaining the official trainer's loss/optimizer loop."""

    def __init__(self, loader, trainer: "PartialSegTrainer", phase: str, every: int):
        self.loader = loader
        self.trainer = trainer
        self.phase = phase
        self.every = every

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        total = len(self)
        started = time.monotonic()
        for index, batch in enumerate(self.loader, start=1):
            self.trainer._progress_batch = index
            self.trainer._progress_total = total
            yield batch
            if index == 1 or index % self.every == 0 or index == total:
                elapsed = time.monotonic() - started
                remaining = elapsed / index * (total - index)
                self.trainer.logger.info(
                    "%s progress: epoch %d/%d, batch %d/%d (%.1f%%), "
                    "loss %.4f, foreground Dice %.3f, LR %s, ETA %s",
                    self.phase, self.trainer.epoch + 1, self.trainer.config.train.epochs,
                    index, total, 100 * index / total,
                    self.trainer._progress_loss, self.trainer._progress_dice,
                    [f"{group['lr']:.2e}" for group in self.trainer.optimizer.param_groups],
                    str(timedelta(seconds=int(remaining))),
                )


class PartialSegTrainer(trainer_package.SegTrainer):
    def make_model(self) -> None:
        super().make_model()
        policy = self.config.partial
        backbone = self.model.backbone
        n_blocks = len(backbone.blocks)
        n_frozen = int(policy.freeze_first_blocks)
        if not 1 <= n_frozen <= n_blocks:
            raise ValueError(f"freeze_first_blocks must be in [1, {n_blocks}]")
        if backbone.use_checkpoint:
            raise ValueError("Disable gradient checkpointing when freezing early blocks")

        # Freeze the pretrained representation, then opt in the late blocks.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for block in backbone.blocks[n_frozen:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

        # The official pretraining checkpoint does not contain SegVit FPN weights.
        for name in ("fpn1", "fpn2", "fpn3", "fpn4"):
            for parameter in getattr(backbone, name).parameters():
                parameter.requires_grad_(True)
        for parameter in self.model.decode_head.parameters():
            parameter.requires_grad_(True)

        counts = {
            "frozen": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        self.logger.info(
            "Partial fine-tuning: frozen ViT blocks [0,%d), trainable blocks [%d,%d), "
            "trainable FPN and decoder; parameters=%s",
            n_frozen, n_frozen, n_blocks, counts,
        )

    def make_optimizer(self) -> None:
        policy = self.config.partial
        if self.config.train.optimizer.name.lower() != "adamw":
            raise ValueError("Partial fine-tuning currently supports AdamW only")
        if self.config.train.accumulation_steps != 1:
            raise ValueError("Use accumulation_steps=1: upstream steps the optimizer every batch")
        if self.config.train.warmup_epochs != 0:
            raise ValueError("Use warmup_epochs=0 to preserve the two learning-rate groups")
        backbone_lr = float(policy.backbone_lr)
        head_lr = float(policy.head_lr)
        if backbone_lr <= 0 or head_lr <= 0:
            raise ValueError("backbone_lr and head_lr must be positive")

        grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = defaultdict(list)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            role = "backbone" if name.startswith("backbone.blocks.") else "head"
            no_decay = parameter.ndim == 1 or name.endswith(".bias")
            grouped[(role, no_decay)].append(parameter)
        if not any(role == "head" for role, _ in grouped):
            raise ValueError("No trainable FPN/decoder parameters found")

        weight_decay = float(self.config.train.weight_decay)
        groups = [
            {
                "params": parameters,
                "lr": backbone_lr if role == "backbone" else head_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            for (role, no_decay), parameters in sorted(grouped.items())
        ]
        optimizer_config = self.config.train.optimizer
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=head_lr,
            eps=float(optimizer_config.eps),
            betas=tuple(float(value) for value in optimizer_config.betas),
        )
        self.lr_scheduler = build_scheduler(
            self.config, self.optimizer, len(self.dataloader_train)
        )
        self.logger.info(
            "Actual optimizer LRs: late backbone=%g, FPN/decoder=%g, weight decay=%g",
            backbone_lr, head_lr, weight_decay,
        )

    def step(self, batch):
        loss, outputs, labels = super().step(batch)
        index = getattr(self, "_progress_batch", 0)
        total = getattr(self, "_progress_total", 0)
        every = int(self.config.partial.log_every)
        if index == 1 or index % every == 0 or index == total:
            with torch.no_grad():
                predicted = outputs.argmax(dim=1) == 1
                target = labels == 1
                overlap = (predicted & target).sum().item()
                area = predicted.sum().item() + target.sum().item()
                self._progress_dice = 1.0 if area == 0 else 2.0 * overlap / area
                self._progress_loss = float(loss.detach().item())
        return loss, outputs, labels

    def train_one_epoch(self, data_loader):
        every = int(self.config.partial.log_every)
        if every < 1:
            raise ValueError("partial.log_every must be positive")
        self.logger.info("Train epoch %d/%d started; %d batches", self.epoch + 1,
                         self.config.train.epochs, len(data_loader))
        return super().train_one_epoch(_ProgressLoader(data_loader, self, "Train", every))

    @torch.no_grad()
    def validate(self, data_loader):
        every = int(self.config.partial.log_every)
        self.logger.info("Validation epoch %d/%d started; %d batches", self.epoch + 1,
                         self.config.train.epochs, len(data_loader))
        return super().validate(_ProgressLoader(data_loader, self, "Validation", every))


@hydra.main(
    config_path="../../external/USFM/configs",
    config_name="train",
    version_base="1.2",
)
def main(config: DictConfig) -> None:
    if config.mode != "train":
        raise ValueError("PartialSegTrainer is for training; use the official entry point for testing")
    PartialSegTrainer(config).fit()


if __name__ == "__main__":
    main()
