from typing import Dict
from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig
from ednaml.config.ExecutionPluginConfig import ExecutionPluginConfig


class ExecutionConfig(BaseConfig):
    OPTIMIZER_BUILDER: str
    MODEL_SERVING: bool
    EPOCHS: int
    SKIPEVAL: bool
    TEST_FREQUENCY: int
    TRAINER: str
    TRAINER_ARGS: Dict[str, str]
    PLUGIN: ExecutionPluginConfig

    def __init__(self, execution_dict, defaults: ConfigDefaults):
        self.OPTIMIZER_BUILDER = execution_dict.get(
            "OPTIMIZER_BUILDER", defaults.OPTIMIZER_BUILDER
        )
        self.MODEL_SERVING = execution_dict.get("MODEL_SERVING", defaults.MODEL_SERVING)
        self.EPOCHS = execution_dict.get("EPOCHS", defaults.EPOCHS)
        self.SKIPEVAL = execution_dict.get("SKIPEVAL", defaults.SKIPEVAL)
        self.TEST_FREQUENCY = execution_dict.get(
            "TEST_FREQUENCY", defaults.TEST_FREQUENCY
        )
        self.FP16 = execution_dict.get("FP16", defaults.FP16)
        self.TRAINER = execution_dict.get("TRAINER", defaults.TRAINER)

        self.TRAINER_ARGS = execution_dict.get("TRAINER_ARGS", defaults.TRAINER_ARGS)
        self.PLUGIN = ExecutionPluginConfig(execution_dict.get("PLUGIN", {}))
