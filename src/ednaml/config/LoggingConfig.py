from typing import Any, Dict
from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class LoggingConfig(BaseConfig):
    STEP_VERBOSE: int
    INPUT_SIZE: list
    LOG_MANAGER: str
    LOG_MANAGER_KWARGS: Dict[str, Any]

    def __init__(self, logging_dict, defaults: ConfigDefaults):
        self.STEP_VERBOSE = logging_dict.get("STEP_VERBOSE", defaults.STEP_VERBOSE)
        self.INPUT_SIZE = logging_dict.get("INPUT_SIZE", defaults.INPUT_SIZE)
        self.LOG_MANAGER = logging_dict.get("LOG_MANAGER", defaults.LOG_MANAGER)
        self.LOG_MANAGER_KWARGS = logging_dict.get("LOG_MANAGER_KWARGS", defaults.LOG_MANAGER_KWARGS)
