from ednaml.config import BaseConfig
from typing import Dict

from ednaml.config.ConfigDefaults import ConfigDefaults


class StorageConfig(BaseConfig):
    STORAGE_NAME: str
    STORAGE_CLASS: str
    STORAGE_URL: str
    STORAGE_ARGS: Dict[str,str]

    def __init__(self, storage_dict, defaults: ConfigDefaults):
        self.STORAGE_NAME = storage_dict.get("STORAGE_NAME", defaults.STORAGE_NAME)
        self.STORAGE_CLASS = storage_dict.get("STORAGE_CLASS", defaults.STORAGE_CLASS)
        self.STORAGE_URL = storage_dict.get("STORAGE_URL", defaults.STORAGE_URL)
        self.STORAGE_ARGS = storage_dict.get("STORAGE_ARGS", defaults.STORAGE_ARGS)
        

