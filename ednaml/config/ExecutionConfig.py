
from ednaml.config.ConfigDefaults import ConfigDefaults
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig

class ExecutionConfig:
    OPTIMIZER_BUILDER: str
    MODEL_SERVING: bool
    EPOCHS: int
    SKIPEVAL: bool
    TEST_FREQUENCY: int
    TRAINER: str
    DATAREADER: ExecutionDatareaderConfig

    def __init__(self, execution_dict, defaults: ConfigDefaults):
        self.OPTIMIZER_BUILDER = execution_dict.get("OPTIMIZER_BUILDER", defaults.OPTIMIZER_BUILDER)
        self.MODEL_SERVING = execution_dict.get("MODEL_SERVING", defaults.MODEL_SERVING)
        self.EPOCHS = execution_dict.get("EPOCHS", defaults.EPOCHS)
        self.SKIPEVAL = execution_dict.get("SKIPEVAL", defaults.SKIPEVAL)
        self.TEST_FREQUENCY = execution_dict.get("TEST_FREQUENCY", defaults.TEST_FREQUENCY)
        self.TRAINER = execution_dict.get("TRAINER", defaults.TRAINER)

        self.DATAREADER = ExecutionDatareaderConfig(execution_dict.get("DATAREADER", {}))