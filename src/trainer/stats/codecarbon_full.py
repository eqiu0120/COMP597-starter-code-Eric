# src/trainer/stats/codecarbon_full.py
"""
Experiment 2: end-to-end energy measurement with a single CodeCarbon tracker.
No per-step or per-substep tasks — just one measurement for the entire run.
measure_power_secs=0.5 samples hardware every 500 ms.
"""
import logging
import os

import torch
from codecarbon import OfflineEmissionsTracker

import src.config as config
import src.trainer.stats.base as base
from src.trainer.stats.codecarbon import SimpleFileOutput

logger = logging.getLogger(__name__)

trainer_stats_name = "codecarbon_full"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", torch.get_default_device())
    c = conf.trainer_stats_configs.codecarbon_full
    return CodeCarbonFullStats(device, c.run_num, c.project_name, c.output_dir)


class CodeCarbonFullStats(base.TrainerStats):
    """Single end-to-end CodeCarbon tracker — Experiment 2.

    Measures total energy/CO2 for the whole training run with 500 ms
    hardware sampling. No per-step or per-substep overhead.
    """

    def __init__(self, device: torch.device, run_num: int,
                 project_name: str, output_dir: str) -> None:
        self.device = device
        gpu_id = device.index if device.index is not None else 0
        run_number = f"run_{run_num}_"
        os.makedirs(output_dir, exist_ok=True)

        self.tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[SimpleFileOutput(
                output_file_name=f"{run_number}cc_full_only_rank_{gpu_id}.csv",
                output_dir=output_dir,
            )],
            allow_multiple_runs=True,
            log_level="warning",
            gpu_ids=[gpu_id],
            measure_power_secs=0.5,
        )

    def _sync(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_train(self):
        self._sync()
        self.tracker.start()

    def stop_train(self):
        self._sync()
        self.tracker.stop()

    def start_step(self): pass
    def stop_step(self): pass
    def start_forward(self): pass
    def stop_forward(self): pass
    def start_backward(self): pass
    def stop_backward(self): pass
    def start_optimizer_step(self): pass
    def stop_optimizer_step(self): pass
    def start_save_checkpoint(self): pass
    def stop_save_checkpoint(self): pass
    def log_step(self): pass
    def log_loss(self, loss: torch.Tensor): pass
    def log_stats(self): pass
