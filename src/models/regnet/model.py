# src/models/regnet/model.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision

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


def build_trainer(conf: config.Config, dataset: data.Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    stats_obj = stats.init_from_conf(conf, device=device)

    trainer = VisionTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        loss_fn=loss_fn,
        stats_obj=stats_obj,
    )
    return trainer, None