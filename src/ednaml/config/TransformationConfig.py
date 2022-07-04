from typing import Any, Dict, List
from ednaml.config import BaseConfig

from ednaml.config.ConfigDefaults import ConfigDefaults


class TransformationConfig(BaseConfig):
    BATCH_SIZE: int
    WORKERS: int
    ARGS: Dict[str,Any]

    def __init__(self, transformation_dict, defaults: ConfigDefaults): #all of this will go, as shape mean etc won't matter
        self.BATCH_SIZE = transformation_dict.get(
            "BATCH_SIZE", defaults.BATCH_SIZE
        )
        self.WORKERS = transformation_dict.get("WORKERS", defaults.WORKERS)
        self.ARGS = transformation_dict.get("ARGS", defaults.TRANSFORM_ARGS)
        
