from typing import Dict

from ednaml.config import BaseConfig


class ExecutionPluginConfig(BaseConfig):
    #EPOCHS: int
    HOOKS: str   # always | warmup | activated
    RESET: bool

    def __init__(self, ep_dict):
        #self.EPOCHS = datareader_dict.get("PLUGIN_EPOCHS", 1)
        self.HOOKS = ep_dict.get("PLUGIN_HOOKS", 'always')
        self.RESET = ep_dict.get("PLUGIN_RESET", False)