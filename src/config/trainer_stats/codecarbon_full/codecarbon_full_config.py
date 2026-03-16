from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_full"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(type=int, default=0,
            help="Run number for output file naming.")
        self._arg_project_name = _Arg(type=str, default="regnet-energy",
            help="CodeCarbon project name.")
        self._arg_output_dir = _Arg(type=str, default=".",
            help="Directory to save the CodeCarbon CSV.")
