import time
import logging

import torch
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "per_step"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    device = kwargs.get("device", torch.get_default_device())
    calibration_steps = conf.trainer_stats_configs.per_step.calibration_steps
    return PerStepTrainerStats(device=device, calibration_steps=calibration_steps)


class PerStepTrainerStats(base.TrainerStats):
    """Per-step only measurement with phase estimation from calibration.

    During the first `calibration_steps` steps, full per-phase syncs are used
    to measure phase time and energy ratios. After calibration, only two syncs
    occur per step (start and end), and per-phase breakdowns are estimated by
    applying the calibration ratios to the per-step total.

    This demonstrates reduced instrumentation overhead compared to the full
    combined (Exp3) setup while still providing estimated per-phase breakdown.
    """

    def __init__(self, device: torch.device, calibration_steps: int = 50) -> None:
        super().__init__()
        self.device = device
        self.calibration_steps = calibration_steps
        self._step_count = 0
        self._in_calibration = True

        # Phase ratio estimates from calibration (time fractions)
        self._ratio_fwd = 1/3
        self._ratio_bwd = 1/3
        self._ratio_opt = 1/3

        # Energy ratio estimates from calibration
        self._energy_ratio_fwd = 1/3
        self._energy_ratio_bwd = 1/3
        self._energy_ratio_opt = 1/3

        # Step-level timers and energy (always active)
        self.step_t = utils.RunningTimer()
        self._train_wall_start_ns = 0
        self._train_wall_elapsed_ns = 0

        # Calibration phase timers
        self._cal_fwd_t = utils.RunningTimer()
        self._cal_bwd_t = utils.RunningTimer()
        self._cal_opt_t = utils.RunningTimer()

        # Estimated phase timers (for post-calibration logging)
        self._est_fwd_ms = utils.RunningStat()
        self._est_bwd_ms = utils.RunningStat()
        self._est_opt_ms = utils.RunningStat()

        # NVML
        self._nvml_ok = False
        self._nvml_handle = None
        self._energy_step = None
        self._cal_energy_fwd = None
        self._cal_energy_bwd = None
        self._cal_energy_opt = None
        self._est_energy_step = utils.RunningStat()

        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = self.device.index if self.device.index is not None else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_ok = True
                self._energy_step = utils.RunningEnergy(idx)
                self._cal_energy_fwd = utils.RunningEnergy(idx)
                self._cal_energy_bwd = utils.RunningEnergy(idx)
                self._cal_energy_opt = utils.RunningEnergy(idx)
            except Exception as e:
                logger.warning(f"NVML unavailable: {e}")

    def _sync(self):
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start_train(self):
        self._train_wall_start_ns = time.perf_counter_ns()

    def stop_train(self):
        self._train_wall_elapsed_ns = time.perf_counter_ns() - self._train_wall_start_ns

    def start_step(self):
        self._sync()
        self.step_t.start()
        if self._energy_step is not None:
            self._energy_step.start()

    def stop_step(self):
        self._sync()
        self.step_t.stop()
        if self._energy_step is not None:
            self._energy_step.stop()

        step_ms = self.step_t.get_last() / 1e6
        e_step = self._energy_step.get_last() if self._energy_step else 0

        if self._in_calibration:
            self._step_count += 1
            if self._step_count >= self.calibration_steps:
                self._finalise_calibration()
        else:
            # Estimate phase breakdowns from calibration ratios
            self._est_fwd_ms.update(step_ms * self._ratio_fwd)
            self._est_bwd_ms.update(step_ms * self._ratio_bwd)
            self._est_opt_ms.update(step_ms * self._ratio_opt)
            self._est_energy_step.update(e_step)

    def _finalise_calibration(self):
        avg_fwd = self._cal_fwd_t.get_average() / 1e6
        avg_bwd = self._cal_bwd_t.get_average() / 1e6
        avg_opt = self._cal_opt_t.get_average() / 1e6
        total = avg_fwd + avg_bwd + avg_opt
        if total > 0:
            self._ratio_fwd = avg_fwd / total
            self._ratio_bwd = avg_bwd / total
            self._ratio_opt = avg_opt / total

        if self._cal_energy_fwd is not None:
            avg_e_fwd = self._cal_energy_fwd.get_average()
            avg_e_bwd = self._cal_energy_bwd.get_average()
            avg_e_opt = self._cal_energy_opt.get_average()
            e_total = avg_e_fwd + avg_e_bwd + avg_e_opt
            if e_total > 0:
                self._energy_ratio_fwd = avg_e_fwd / e_total
                self._energy_ratio_bwd = avg_e_bwd / e_total
                self._energy_ratio_opt = avg_e_opt / e_total

        self._in_calibration = False
        print(f"\n[per_step] Calibration done ({self.calibration_steps} steps). "
              f"Phase ratios — fwd:{self._ratio_fwd:.2%} "
              f"bwd:{self._ratio_bwd:.2%} "
              f"opt:{self._ratio_opt:.2%}")

    # During calibration: full per-phase syncs
    def start_forward(self):
        if self._in_calibration:
            self._sync()
            self._cal_fwd_t.start()
            if self._cal_energy_fwd is not None:
                self._cal_energy_fwd.start()

    def stop_forward(self):
        if self._in_calibration:
            self._sync()
            self._cal_fwd_t.stop()
            if self._cal_energy_fwd is not None:
                self._cal_energy_fwd.stop()

    def start_backward(self):
        if self._in_calibration:
            self._sync()
            self._cal_bwd_t.start()
            if self._cal_energy_bwd is not None:
                self._cal_energy_bwd.start()

    def stop_backward(self):
        if self._in_calibration:
            self._sync()
            self._cal_bwd_t.stop()
            if self._cal_energy_bwd is not None:
                self._cal_energy_bwd.stop()

    def start_optimizer_step(self):
        if self._in_calibration:
            self._sync()
            self._cal_opt_t.start()
            if self._cal_energy_opt is not None:
                self._cal_energy_opt.start()

    def stop_optimizer_step(self):
        if self._in_calibration:
            self._sync()
            self._cal_opt_t.stop()
            if self._cal_energy_opt is not None:
                self._cal_energy_opt.stop()

    def start_save_checkpoint(self): pass
    def stop_save_checkpoint(self): pass
    def log_loss(self, loss): pass

    def log_step(self):
        step_ms = self.step_t.get_last() / 1e6
        if self._in_calibration:
            fwd_ms = self._cal_fwd_t.get_last() / 1e6
            bwd_ms = self._cal_bwd_t.get_last() / 1e6
            opt_ms = self._cal_opt_t.get_last() / 1e6
            tag = "[cal]"
        else:
            fwd_ms = step_ms * self._ratio_fwd
            bwd_ms = step_ms * self._ratio_bwd
            opt_ms = step_ms * self._ratio_opt
            tag = "[est]"

        e_str = ""
        if self._energy_step is not None:
            e = self._energy_step.get_last()
            e_str = (f" -- energy_step(mJ) {e:.1f}"
                     f" -- energy_fwd(mJ) {e * self._energy_ratio_fwd:.1f}"
                     f" -- energy_bwd(mJ) {e * self._energy_ratio_bwd:.1f}"
                     f" -- energy_opt(mJ) {e * self._energy_ratio_opt:.1f}")

        print(f"{tag} step {step_ms:.4f} -- "
              f"forward {fwd_ms:.4f} -- "
              f"backward {bwd_ms:.4f} -- "
              f"optimizer step {opt_ms:.4f}"
              + e_str)

    def log_stats(self):
        wall_s = self._train_wall_elapsed_ns / 1e9
        n_steps = len(self.step_t.stat.history)
        avg_step = self.step_t.get_average() / 1e6

        cal_fwd = self._cal_fwd_t.get_average() / 1e6
        cal_bwd = self._cal_bwd_t.get_average() / 1e6
        cal_opt = self._cal_opt_t.get_average() / 1e6

        est_fwd = avg_step * self._ratio_fwd
        est_bwd = avg_step * self._ratio_bwd
        est_opt = avg_step * self._ratio_opt

        print("\n###############  SUSTAINABILITY SUMMARY (per_step)  ###############")
        print(f"  Total training wall time : {wall_s:.2f} s")
        print(f"  Steps completed          : {n_steps}")
        print(f"  Calibration steps        : {self.calibration_steps}")
        if wall_s > 0:
            print(f"  Avg throughput           : {n_steps / wall_s:.3f} steps/s")
        print(f"\n  Calibration phase ratios (first {self.calibration_steps} steps, full sync):")
        print(f"    forward   : {self._ratio_fwd:.2%}  ({cal_fwd:.2f} ms avg)")
        print(f"    backward  : {self._ratio_bwd:.2%}  ({cal_bwd:.2f} ms avg)")
        print(f"    optimizer : {self._ratio_opt:.2%}  ({cal_opt:.2f} ms avg)")
        print(f"\n  Post-calibration estimates (full run avg step = {avg_step:.2f} ms):")
        print(f"    forward   : {est_fwd:.2f} ms")
        print(f"    backward  : {est_bwd:.2f} ms")
        print(f"    optimizer : {est_opt:.2f} ms")
        if self._energy_step is not None:
            avg_e = self._energy_step.get_average()
            print(f"\n  Avg GPU energy per step  : {avg_e:.2f} mJ (NVML direct)")
            print(f"    forward   : {avg_e * self._energy_ratio_fwd:.2f} mJ (estimated)")
            print(f"    backward  : {avg_e * self._energy_ratio_bwd:.2f} mJ (estimated)")
            print(f"    optimizer : {avg_e * self._energy_ratio_opt:.2f} mJ (estimated)")
        print("####################################################################\n")
