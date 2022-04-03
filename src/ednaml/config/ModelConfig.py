from typing import Dict

from ednaml.config.ConfigDefaults import ConfigDefaults


class ModelConfig:
    BUILDER: str
    MODEL_ARCH: str
    MODEL_BASE: str
    MODEL_NORMALIZATION: str
    MODEL_KWARGS: Dict[str, str]

    def __init__(self, model_dict, defaults: ConfigDefaults):
        self.BUILDER = model_dict.get("BUILDER", defaults.BUILDER)
        self.MODEL_ARCH = model_dict.get("MODEL_ARCH", defaults.MODEL_ARCH)
        self.MODEL_BASE = model_dict.get("MODEL_BASE", defaults.MODEL_BASE)
        self.MODEL_NORMALIZATION = model_dict.get(
            "MODEL_NORMALIZATION", defaults.MODEL_NORMALIZATION
        )
        self.MODEL_KWARGS = model_dict.get("MODEL_KWARGS", defaults.MODEL_KWARGS)
