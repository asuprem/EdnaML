from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig
from typing import Dict

class DeploymentConfig(BaseConfig):
    OUTPUT_ARGS: Dict[str,str]
    DEPLOYMENT: str

    def __init__(self, deployment_dict, defaults: ConfigDefaults):
        self.OUTPUT_ARGS = deployment_dict.get(
            "OUTPUT_ARGS", defaults.OUTPUT_ARGS
        )
        self.DEPLOYMENT = deployment_dict.get(
            "DEPLOYMENT", defaults.DEPLOYMENT
        )