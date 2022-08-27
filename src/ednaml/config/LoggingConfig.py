from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class LoggingConfig(BaseConfig):
    STEP_VERBOSE: int

    def __init__(self, logging_dict, defaults: ConfigDefaults):
        self.STEP_VERBOSE = logging_dict.get(
            "STEP_VERBOSE", defaults.STEP_VERBOSE
        )
        self.INPUT_SIZE = logging_dict.get(
            "INPUT_SIZE", defaults.INPUT_SIZE
        )
