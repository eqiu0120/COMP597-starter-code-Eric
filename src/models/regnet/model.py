# src/models/regnet/model.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import tqdm.auto

import src.config as config
import src.trainer.stats as stats
from src.trainer.vision import VisionTrainer


def build_model(pretrained: bool = False) -> nn.Module:
    if pretrained:
        weights = torchvision.models.RegNet_Y_128GF_Weights.DEFAULT
    else:
        weights = None

    try:
        model = torchvision.models.regnet_y_128gf(weights=weights)
    except TypeError:
        model = torchvision.models.regnet_y_128gf(pretrained=bool(pretrained))
    return model


class TimedVisionTrainer(VisionTrainer):
    """VisionTrainer that repeats epochs until a wall-clock duration is reached."""

    def __init__(self, duration_seconds: int, **kwargs):
        super().__init__(**kwargs)
        self.duration_seconds = duration_seconds

    def train(self, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        t_start = time.monotonic()
        deadline = t_start + self.duration_seconds
        step = 0
        progress_bar = tqdm.auto.tqdm(desc="loss: N/A")

        self.stats.start_train()
        while time.monotonic() < deadline:
            for batch in self.loader:
                if time.monotonic() >= deadline:
                    break

                self.stats.start_step()
                loss, _ = self.step(step, batch, model_kwargs)
                self.stats.stop_step()

                self.stats.log_loss(loss)
                self.stats.log_step()

                progress_bar.set_description(f"loss: {loss.item():.4f}")
                progress_bar.update(1)
                step += 1

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()

        # Always print a summary line so every experiment (including noop) has parseable output
        elapsed = time.monotonic() - t_start
        bs = self.loader.batch_size
        tp = step * bs / elapsed if elapsed > 0 else 0.0
        print(f"TRAINING_SUMMARY total_steps={step} elapsed_s={elapsed:.1f} "
              f"batch_size={bs} throughput_samples_per_sec={tp:.2f}")


def build_trainer(conf: config.Config, dataset: data.Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read from config if available, else use defaults
    model_conf = getattr(conf.model_configs, "regnet", None)
    batch_size = getattr(model_conf, "batch_size", 8) if model_conf else 8
    duration_seconds = getattr(model_conf, "duration_seconds", 300) if model_conf else 300

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=False).to(device)
    print("REGNET DEBUG device:", device)
    print("REGNET DEBUG model param device:", next(model.parameters()).device)
    print(f"REGNET CONFIG batch_size={batch_size} duration_seconds={duration_seconds}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    stats_obj = stats.init_from_conf(conf, device=device)

    trainer = TimedVisionTrainer(
        duration_seconds=duration_seconds,
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        loss_fn=loss_fn,
        stats_obj=stats_obj,
    )
    return trainer, None