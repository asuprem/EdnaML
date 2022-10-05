from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig
from ednaml.config.ExecutionPluginConfig import ExecutionPluginConfig
from typing import Dict

class DeploymentConfig(BaseConfig):
    OUTPUT_ARGS: Dict[str,str]
    DEPLOYMENT_ARGS: Dict[str,str]
    DEPLOY: str
    DATAREADER: ExecutionDatareaderConfig
    EPOCHS: int
    PLUGIN: ExecutionPluginConfig


    def __init__(self, deployment_dict, defaults: ConfigDefaults):
        self.OUTPUT_ARGS = deployment_dict.get(
            "OUTPUT_ARGS", defaults.OUTPUT_ARGS
        )
        self.DEPLOYMENT_ARGS = deployment_dict.get(
            "DEPLOYMENT_ARGS", defaults.DEPLOYMENT_ARGS
        )
        self.DEPLOY = deployment_dict.get(
            "DEPLOY", defaults.DEPLOY
        )
        self.EPOCHS = deployment_dict.get("EPOCHS", defaults.DEPLOYMENT_EPOCHS)
        self.PLUGIN = ExecutionPluginConfig(deployment_dict.get("PLUGIN", {}))