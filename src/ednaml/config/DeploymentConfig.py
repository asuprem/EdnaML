from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig
from typing import Dict

class DeploymentConfig(BaseConfig):
    OUTPUT_ARGS: Dict[str,str]
    DEPLOYMENT_ARGS: Dict[str,str]
    DEPLOY: str
    DATAREADER: ExecutionDatareaderConfig

    def __init__(self, deployment_dict, defaults: ConfigDefaults):
        self.OUTPUT_ARGS = deployment_dict.get(
            "OUTPUT_ARGS", defaults.OUTPUT_ARGS
        )
        self.DEPLOYMENT_ARGS = deployment_dict.get(
            "DEPLOYMENT_ARGS", defaults.DEPLOYMENT_ARGS
        )
        self.DEPLOYMENT = deployment_dict.get(
            "DEPLOYMENT", defaults.DEPLOYMENT
        )
        self.DATAREADER = ExecutionDatareaderConfig(
            deployment_dict.get("DATAREADER", {})
        )