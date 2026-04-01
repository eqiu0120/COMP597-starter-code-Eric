from src.config.util.base_config import _Arg, _BaseConfig

config_name = "per_step"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_calibration_steps = _Arg(type=int, help="Number of steps with full per-phase sync for calibration.", default=50)
