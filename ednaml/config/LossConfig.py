from typing import Dict, List

from ednaml.config.ConfigDefaults import ConfigDefaults


class LossConfig:
    LOSSES: List[str]
    KWARGS: List[Dict[str,str]]
    LAMBDAS: List[int]
    LABEL: str
    NAME: str

    def __init__(self, loss_dict, defaults: ConfigDefaults):
        self.LOSSES = loss_dict.get("LOSSES", defaults.LOSSES)
        self.KWARGS = loss_dict.get("KWARGS", defaults.LOSS_KWARGS)
        self.LAMBDAS = loss_dict.get("LAMBDAS", defaults.LAMBDAS)
        self.LABEL = loss_dict.get("LABEL", defaults.LOSS_LABEL)
        self.NAME = loss_dict.get("NAME", defaults.LOSS_NAME)