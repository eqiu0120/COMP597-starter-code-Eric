import logging
import os
import torch
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
from src.trainer.stats.resource import ResourceTrainerStats
from src.trainer.stats.codecarbon import CodeCarbonStats

logger = logging.getLogger(__name__)

trainer_stats_name = "combined"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", torch.get_default_device())
    cc_conf = conf.trainer_stats_configs.combined
    return CombinedTrainerStats(
        device=device,
        run_num=cc_conf.run_num,
        project_name=cc_conf.project_name,
        output_dir=cc_conf.output_dir,
    )


class CombinedTrainerStats(base.TrainerStats):
    """Combines ResourceTrainerStats and CodeCarbonStats into one.

    Provides simultaneous per-step GPU resource measurements (utilization,
    memory, I/O, timing) and energy / carbon-emission tracking via CodeCarbon.

    Synchronization
    ---------------
    CUDA kernels are asynchronous: Python returns immediately while the GPU
    is still executing.  To get accurate wall-clock measurements we must call
    ``torch.cuda.synchronize`` *before* recording a timestamp so that all
    outstanding GPU work is guaranteed to have finished.

    Each inner stats class (ResourceTrainerStats, CodeCarbonStats) contains its
    own ``torch.cuda.synchronize`` call.  To avoid double-syncing (which is
    harmless but wasteful), ``CombinedTrainerStats`` calls ``_sync`` once at
    each measurement boundary — *before* delegating to either inner class.
    The subsequent inner syncs then find the GPU already idle and return
    immediately.

    Parameters
    ----------
    device
        The CUDA device used for training.
    run_num
        Run identifier forwarded to CodeCarbon for file naming.
    project_name
        Project name forwarded to CodeCarbon.
    output_dir
        Directory where CodeCarbon CSV files are written.
    """

    def __init__(
        self,
        device: torch.device,
        run_num: int,
        project_name: str,
        output_dir: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.resource = ResourceTrainerStats(device=device)
        self.carbon = CodeCarbonStats(device, run_num, project_name, output_dir)

    def _sync(self) -> None:
        """Synchronise the CUDA device once before starting/stopping measurements.

        Both inner stats classes also call synchronize internally; because the
        GPU is already idle after this call those become no-ops, ensuring
        exactly one blocking sync per measurement boundary.
        """
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ #
    # TrainerStats interface                                               #
    # ------------------------------------------------------------------ #

    def start_train(self) -> None:
        self._sync()
        self.resource.start_train()
        self.carbon.start_train()

    def stop_train(self) -> None:
        self._sync()
        self.resource.stop_train()
        self.carbon.stop_train()

    def start_step(self) -> None:
        self._sync()
        self.resource.start_step()
        self.carbon.start_step()

    def stop_step(self) -> None:
        self._sync()
        self.resource.stop_step()
        self.carbon.stop_step()

    def start_forward(self) -> None:
        self._sync()
        self.resource.start_forward()
        self.carbon.start_forward()

    def stop_forward(self) -> None:
        self._sync()
        self.resource.stop_forward()
        self.carbon.stop_forward()

    def start_backward(self) -> None:
        self._sync()
        self.resource.start_backward()
        self.carbon.start_backward()

    def stop_backward(self) -> None:
        self._sync()
        self.resource.stop_backward()
        self.carbon.stop_backward()

    def start_optimizer_step(self) -> None:
        self._sync()
        self.resource.start_optimizer_step()
        self.carbon.start_optimizer_step()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self.resource.stop_optimizer_step()
        self.carbon.stop_optimizer_step()

    def start_save_checkpoint(self) -> None:
        self._sync()
        self.resource.start_save_checkpoint()
        self.carbon.start_save_checkpoint()

    def stop_save_checkpoint(self) -> None:
        self._sync()
        self.resource.stop_save_checkpoint()
        self.carbon.stop_save_checkpoint()

    def log_loss(self, loss: torch.Tensor) -> None:
        self.resource.log_loss(loss)
        self.carbon.log_loss(loss)

    def log_step(self) -> None:
        # Only resource emits per-step stdout; CodeCarbon writes CSV at stop_step.
        self.resource.log_step()

    def log_stats(self) -> None:
        print("############### RESOURCE STATS ###############")
        self.resource.log_stats()
        print("############### CODECARBON STATS ###############")
        self.carbon.log_stats()
