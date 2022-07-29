from ednaml.config import BaseConfig
from typing import Dict

from ednaml.config.ConfigDefaults import ConfigDefaults


class StorageConfig(BaseConfig):
    TYPE: str
    STORAGE_ARGS: Dict
    URL: str

    def __init__(self, storage_dict,defaults: ConfigDefaults):
        self.TYPE = storage_dict.get("TYPE", defaults.STORAGE_TYPE)
        self.STORAGE_ARGS = storage_dict.get("STORAGE_ARGS", defaults.STORAGE_ARGS)
        self.URL = storage_dict.get("URL", defaults.STORAGE_URL)
        

