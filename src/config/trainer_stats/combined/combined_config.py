from src.config.util.base_config import _Arg, _BaseConfig

config_name = "combined"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(type=int, help="Run number used for CodeCarbon file naming.", default=0)
        self._arg_project_name = _Arg(type=str, help="Project name used by CodeCarbon.", default="regnet-energy")
        self._arg_output_dir = _Arg(type=str, help="Directory where CodeCarbon CSV files are written.", default=".")
