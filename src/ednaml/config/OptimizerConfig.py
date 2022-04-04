from typing import Dict

from ednaml.config.ConfigDefaults import ConfigDefaults


class OptimizerConfig:
    OPTIMIZER_NAME: str
    OPTIMIZER: str
    OPTIMIZER_KWARGS: Dict[str, str]
    BASE_LR: float
    LR_BIAS_FACTOR: float
    WEIGHT_DECAY: float
    WEIGHT_BIAS_FACTOR: float

    def __init__(self, optimizer_dict, defaults: ConfigDefaults):
        self.OPTIMIZER_NAME = optimizer_dict.get(
            "OPTIMIZER_NAME", defaults.OPTIMIZER_NAME
        )
        self.OPTIMIZER = optimizer_dict.get("OPTIMIZER", defaults.OPTIMIZER)
        self.OPTIMIZER_KWARGS = optimizer_dict.get(
            "OPTIMIZER_KWARGS", defaults.OPTIMIZER_KWARGS
        )
        self.BASE_LR = optimizer_dict.get("BASE_LR", defaults.BASE_LR)
        self.LR_BIAS_FACTOR = optimizer_dict.get(
            "LR_BIAS_FACTOR", defaults.LR_BIAS_FACTOR
        )
        self.WEIGHT_DECAY = optimizer_dict.get("WEIGHT_DECAY", defaults.WEIGHT_DECAY)
        self.WEIGHT_BIAS_FACTOR = optimizer_dict.get(
            "WEIGHT_BIAS_FACTOR", defaults.WEIGHT_BIAS_FACTOR
        )
