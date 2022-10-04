from typing import Dict
from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class ModelPluginConfig(BaseConfig):
    PLUGIN: str
    PLUGIN_NAME: str
    PLUGIN_KWARGS: Dict[str, str]
    
    
    def __init__(self, save_dict, defaults: ConfigDefaults):
        self.PLUGIN = save_dict.get(
            "PLUGIN", defaults.PLUGIN
        )
        self.PLUGIN_NAME = save_dict.get(
            "PLUGIN_NAME", defaults.PLUGIN_NAME
        )
        self.PLUGIN_KWARGS = save_dict.get(
            "PLUGIN_KWARGS", defaults.PLUGIN_KWARGS
        )
        