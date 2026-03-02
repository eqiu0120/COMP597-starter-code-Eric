import os
import logging
from dataclasses import dataclass

import torch
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "resource"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", torch.get_default_device())
    return ResourceTrainerStats(device=device)

def _read_proc_io_bytes() -> tuple[int, int]:
    """
    Read process I/O from /proc/self/io (Linux).
    Returns (read_bytes, write_bytes). If unavailable, returns (0, 0).
    """
    try:
        rb = wb = 0
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    rb = int(line.split(":")[1].strip())
                elif line.startswith("write_bytes:"):
                    wb = int(line.split(":")[1].strip())
        return rb, wb
    except Exception:
        return 0, 0

@dataclass
class _GpuSample:
    util: float = 0.0       # %
    mem_used_mb: float = 0.0

class ResourceTrainerStats(base.TrainerStats):
    """
    Extends Simple timing with:
      - GPU util (%) + GPU memory used (MB) per step 
      - Process IO delta read/write bytes per step
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Timing (same style as SimpleTrainerStats)
        self.step_t = utils.RunningTimer()
        self.fwd_t = utils.RunningTimer()
        self.bwd_t = utils.RunningTimer()
        self.opt_t = utils.RunningTimer()

        # Resource stats
        self.gpu_util = utils.RunningStat()
        self.gpu_mem_mb = utils.RunningStat()
        self.io_read_bytes = utils.RunningStat()
        self.io_write_bytes = utils.RunningStat()

        self._io_start = (0, 0)

        # NVML init lazily (only if CUDA device)
        self._nvml_ok = False
        self._nvml_handle = None
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = self.device.index if self.device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_ok = True
            except Exception as e:
                logger.warning(f"NVML unavailable; GPU util/mem will be skipped. ({e})")

    def _sync(self):
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _sample_gpu(self) -> _GpuSample:
        if not self._nvml_ok:
            return _GpuSample()
        import pynvml
        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        mem_used_mb = mem.used / (1024 * 1024)
        return _GpuSample(util=float(util), mem_used_mb=float(mem_used_mb))

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        pass

    def start_step(self) -> None:
        self._sync()
        self.step_t.start()
        self._io_start = _read_proc_io_bytes()

    def stop_step(self) -> None:
        self._sync()
        self.step_t.stop()

        # GPU sample at end of step
        g = self._sample_gpu()
        if g.util > 0 or g.mem_used_mb > 0:
            self.gpu_util.update(g.util)
            self.gpu_mem_mb.update(g.mem_used_mb)

        # IO delta over the step
        rb0, wb0 = self._io_start
        rb1, wb1 = _read_proc_io_bytes()
        self.io_read_bytes.update(max(0, rb1 - rb0))
        self.io_write_bytes.update(max(0, wb1 - wb0))

    def start_forward(self) -> None:
        self._sync()
        self.fwd_t.start()

    def stop_forward(self) -> None:
        self._sync()
        self.fwd_t.stop()

    def start_backward(self) -> None:
        self._sync()
        self.bwd_t.start()

    def stop_backward(self) -> None:
        self._sync()
        self.bwd_t.stop()

    def start_optimizer_step(self) -> None:
        self._sync()
        self.opt_t.start()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self.opt_t.stop()

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        # Same style as SimpleTrainerStats printing ms
        print(
            f"step {self.step_t.get_last()/1e6:.4f} -- "
            f"forward {self.fwd_t.get_last()/1e6:.4f} -- "
            f"backward {self.bwd_t.get_last()/1e6:.4f} -- "
            f"optimizer step {self.opt_t.get_last()/1e6:.4f} -- "
            f"gpu_util% {self.gpu_util.get_last():.2f} -- "
            f"gpu_mem(MB) {self.gpu_mem_mb.get_last():.1f} -- "
            f"io_read(B) {int(self.io_read_bytes.get_last())} -- "
            f"io_write(B) {int(self.io_write_bytes.get_last())}"
        )

    def log_stats(self) -> None:
        # Time averages (ms)
        print(
            f"AVG : step {self.step_t.get_average()/1e6:.4f} -- "
            f"forward {self.fwd_t.get_average()/1e6:.4f} -- "
            f"backward {self.bwd_t.get_average()/1e6:.4f} -- "
            f"optimizer step {self.opt_t.get_average()/1e6:.4f}"
        )

        print("###############        Step        ###############")
        self.step_t.log_analysis()
        print("###############      FORWARD       ###############")
        self.fwd_t.log_analysis()
        print("###############      BACKWARD      ###############")
        self.bwd_t.log_analysis()
        print("###############   OPTIMIZER STEP   ###############")
        self.opt_t.log_analysis()

        # Resource summaries
        print("###############    GPU UTIL (%)    ###############")
        self.gpu_util.log_analysis()
        print("###############   GPU MEM (MB)     ###############")
        self.gpu_mem_mb.log_analysis()
        print("###############   IO READ (B)      ###############")
        self.io_read_bytes.log_analysis()
        print("###############   IO WRITE (B)     ###############")
        self.io_write_bytes.log_analysis()
