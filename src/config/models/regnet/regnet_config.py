from src.config.util.base_config import _Arg, _BaseConfig

config_name = "regnet"

class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_batch_size = _Arg(type=int, default=8,
            help="Mini-batch size for training.")
        self._arg_duration_seconds = _Arg(type=int, default=300,
            help="Wall-clock seconds to train for (default 300 = 5 min).")
