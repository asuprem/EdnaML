from ednaml.config.ConfigDefaults import ConfigDefaults


class LoggingConfig:
    STEP_VERBOSE: int

    def __init__(self, logging_dict, defaults: ConfigDefaults):
        self.STEP_VERBOSE = logging_dict.get("STEP_VERBOSE", defaults.STEP_VERBOSE)
