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
      - Direct NVML GPU energy (mJ) per step, forward, backward, and optimizer
        substep — this is the raw hardware counter, more granular than CodeCarbon
      - Total training wall-clock time (start_train -> stop_train)
      - Data loading overhead derived from step - (fwd + bwd + opt)
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Timing
        self.step_t = utils.RunningTimer()
        self.fwd_t  = utils.RunningTimer()
        self.bwd_t  = utils.RunningTimer()
        self.opt_t  = utils.RunningTimer()

        # Total training wall-clock timer (start_train -> stop_train)
        self._train_wall_start_ns: int = 0
        self._train_wall_elapsed_ns: int = 0

        # Resource stats
        self.gpu_util      = utils.RunningStat()
        self.gpu_mem_mb    = utils.RunningStat()
        self.io_read_bytes  = utils.RunningStat()
        self.io_write_bytes = utils.RunningStat()

        self._io_start = (0, 0)

        # NVML init (only if CUDA device)
        self._nvml_ok     = False
        self._nvml_handle = None
        self._energy_step: utils.RunningEnergy | None = None
        self._energy_fwd:  utils.RunningEnergy | None = None
        self._energy_bwd:  utils.RunningEnergy | None = None
        self._energy_opt:  utils.RunningEnergy | None = None

        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = self.device.index if self.device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_ok = True
                # RunningEnergy uses the same NVML handle internally
                self._energy_step = utils.RunningEnergy(idx)
                self._energy_fwd  = utils.RunningEnergy(idx)
                self._energy_bwd  = utils.RunningEnergy(idx)
                self._energy_opt  = utils.RunningEnergy(idx)
            except Exception as e:
                logger.warning(f"NVML unavailable; GPU util/mem/energy will be skipped. ({e})")

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
        import time
        self._train_wall_start_ns = time.perf_counter_ns()

    def stop_train(self) -> None:
        import time
        self._train_wall_elapsed_ns = time.perf_counter_ns() - self._train_wall_start_ns

    def start_step(self) -> None:
        self._sync()
        self.step_t.start()
        self._io_start = _read_proc_io_bytes()
        if self._energy_step is not None:
            self._energy_step.start()

    def stop_step(self) -> None:
        self._sync()
        self.step_t.stop()
        if self._energy_step is not None:
            self._energy_step.stop()

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
        if self._energy_fwd is not None:
            self._energy_fwd.start()

    def stop_forward(self) -> None:
        self._sync()
        self.fwd_t.stop()
        if self._energy_fwd is not None:
            self._energy_fwd.stop()

    def start_backward(self) -> None:
        self._sync()
        self.bwd_t.start()
        if self._energy_bwd is not None:
            self._energy_bwd.start()

    def stop_backward(self) -> None:
        self._sync()
        self.bwd_t.stop()
        if self._energy_bwd is not None:
            self._energy_bwd.stop()

    def start_optimizer_step(self) -> None:
        self._sync()
        self.opt_t.start()
        if self._energy_opt is not None:
            self._energy_opt.start()

    def stop_optimizer_step(self) -> None:
        self._sync()
        self.opt_t.stop()
        if self._energy_opt is not None:
            self._energy_opt.stop()

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        step_ms  = self.step_t.get_last() / 1e6
        fwd_ms   = self.fwd_t.get_last()  / 1e6
        bwd_ms   = self.bwd_t.get_last()  / 1e6
        opt_ms   = self.opt_t.get_last()  / 1e6
        # Data loading = step wall time minus the three measured substeps.
        # This is the time the GPU sat idle waiting for the next batch.
        data_ms  = max(0.0, step_ms - fwd_ms - bwd_ms - opt_ms)

        energy_parts = ""
        if self._energy_step is not None:
            energy_parts = (
                f" -- energy_step(mJ) {self._energy_step.get_last():.1f}"
                f" -- energy_fwd(mJ) {self._energy_fwd.get_last():.1f}"
                f" -- energy_bwd(mJ) {self._energy_bwd.get_last():.1f}"
                f" -- energy_opt(mJ) {self._energy_opt.get_last():.1f}"
            )

        print(
            f"step {step_ms:.4f} -- "
            f"forward {fwd_ms:.4f} -- "
            f"backward {bwd_ms:.4f} -- "
            f"optimizer step {opt_ms:.4f} -- "
            f"data_loading {data_ms:.4f} -- "
            f"gpu_util% {self.gpu_util.get_last():.2f} -- "
            f"gpu_mem(MB) {self.gpu_mem_mb.get_last():.1f} -- "
            f"io_read(B) {int(self.io_read_bytes.get_last())} -- "
            f"io_write(B) {int(self.io_write_bytes.get_last())}"
            + energy_parts
        )

    def log_stats(self) -> None:
        avg_step = self.step_t.get_average() / 1e6
        avg_fwd  = self.fwd_t.get_average()  / 1e6
        avg_bwd  = self.bwd_t.get_average()  / 1e6
        avg_opt  = self.opt_t.get_average()  / 1e6
        avg_data = max(0.0, avg_step - avg_fwd - avg_bwd - avg_opt)

        print(
            f"AVG : step {avg_step:.4f} -- "
            f"forward {avg_fwd:.4f} -- "
            f"backward {avg_bwd:.4f} -- "
            f"optimizer step {avg_opt:.4f} -- "
            f"data_loading {avg_data:.4f} (all ms)"
        )

        # ---- Sustainability summary ----------------------------------------
        wall_s = self._train_wall_elapsed_ns / 1e9
        n_steps = len(self.step_t.stat.history)
        print("\n###############  SUSTAINABILITY SUMMARY  ###############")
        print(f"  Total training wall time : {wall_s:.2f} s")
        print(f"  Steps completed          : {n_steps}")
        if wall_s > 0:
            print(f"  Avg throughput           : {n_steps / wall_s:.3f} steps/s")
        if avg_step > 0:
            pct_fwd  = avg_fwd  / avg_step * 100
            pct_bwd  = avg_bwd  / avg_step * 100
            pct_opt  = avg_opt  / avg_step * 100
            pct_data = avg_data / avg_step * 100
            print(f"  Step time breakdown:")
            print(f"    forward      : {pct_fwd:.1f}%")
            print(f"    backward     : {pct_bwd:.1f}%")
            print(f"    optimizer    : {pct_opt:.1f}%")
            print(f"    data loading : {pct_data:.1f}%  <-- GPU idle waiting for data")
        if self._energy_step is not None:
            avg_e_step = self._energy_step.get_average()
            avg_e_fwd  = self._energy_fwd.get_average()
            avg_e_bwd  = self._energy_bwd.get_average()
            avg_e_opt  = self._energy_opt.get_average()
            print(f"  Avg GPU energy per step  : {avg_e_step:.2f} mJ (NVML direct)")
            if avg_e_step > 0:
                print(f"  GPU energy breakdown:")
                print(f"    forward      : {avg_e_fwd / avg_e_step * 100:.1f}%  ({avg_e_fwd:.2f} mJ)")
                print(f"    backward     : {avg_e_bwd / avg_e_step * 100:.1f}%  ({avg_e_bwd:.2f} mJ)")
                print(f"    optimizer    : {avg_e_opt / avg_e_step * 100:.1f}%  ({avg_e_opt:.2f} mJ)")
        print("########################################################\n")

        # ---- Per-metric breakdowns ----------------------------------------
        print("###############        Step        ###############")
        self.step_t.log_analysis()
        print("###############      FORWARD       ###############")
        self.fwd_t.log_analysis()
        print("###############      BACKWARD      ###############")
        self.bwd_t.log_analysis()
        print("###############   OPTIMIZER STEP   ###############")
        self.opt_t.log_analysis()
        print("###############    GPU UTIL (%)    ###############")
        self.gpu_util.log_analysis()
        print("###############   GPU MEM (MB)     ###############")
        self.gpu_mem_mb.log_analysis()
        print("###############   IO READ (B)      ###############")
        self.io_read_bytes.log_analysis()
        print("###############   IO WRITE (B)     ###############")
        self.io_write_bytes.log_analysis()
        if self._energy_step is not None:
            print("###############  GPU ENERGY/STEP (mJ) NVML  ###############")
            self._energy_step.log_analysis()
            print("###############  GPU ENERGY/FWD  (mJ) NVML  ###############")
            self._energy_fwd.log_analysis()
            print("###############  GPU ENERGY/BWD  (mJ) NVML  ###############")
            self._energy_bwd.log_analysis()
            print("###############  GPU ENERGY/OPT  (mJ) NVML  ###############")
            self._energy_opt.log_analysis()
