# src/trainer/vision.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import src.trainer.base as base
import src.trainer.stats as stats


class VisionTrainer(base.Trainer):
    """
    Minimal trainer for torchvision-style classification models.

    Expects each batch to be either:
      - a tuple: (images, labels)
      - OR a dict containing keys like {"images"/"pixel_values", "labels"}
    """

    def __init__(
        self,
        loader: data.DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        device: torch.device,
        loss_fn: nn.Module,
        stats_obj: stats.TrainerStats = stats.NOOPTrainerStats(),
    ):
        super().__init__(model=model, loader=loader, device=device, stats=stats_obj)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

    def process_batch(self, i: int, batch: Any) -> Any:
        # Override base.process_batch (which assumes dict-of-tensors).
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            return x.to(self.device), y.to(self.device)

        if isinstance(batch, dict):
            # Try common keys
            x = batch.get("images", batch.get("pixel_values", None))
            y = batch.get("labels", None)
            if x is None or y is None:
                raise ValueError(f"VisionTrainer expected dict with images/pixel_values and labels keys, got {batch.keys()}")
            return x.to(self.device), y.to(self.device)

        raise ValueError(f"Unsupported batch type: {type(batch)}")

    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        x, y = batch
        logits = self.model(x, **model_kwargs)
        loss = self.loss_fn(logits, y)
        return loss

    def backward(self, i: int, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, i: int) -> None:
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
