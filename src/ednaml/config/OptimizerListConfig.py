from ednaml.config import BaseConfig
from ednaml.config.OptimizerConfig import OptimizerConfig


class OptimizerListConfig(BaseConfig):
    OPTIMIZER: OptimizerConfig

    def __init__(self):
        pass
