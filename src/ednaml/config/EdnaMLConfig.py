# BaseConfig class to manage the input configurations. TODO
import os
import json
from typing import Dict, List
import yaml
from ednaml.config import BaseConfig
from ednaml.config.DeploymentConfig import DeploymentConfig
from ednaml.utils import config_serializer
from ednaml.config.ConfigDefaults import ConfigDefaults

from ednaml.config.ExecutionConfig import ExecutionConfig
from ednaml.config.LoggingConfig import LoggingConfig
from ednaml.config.LossConfig import LossConfig
from ednaml.config.OptimizerConfig import OptimizerConfig
from ednaml.config.SaveConfig import SaveConfig
from ednaml.config.SchedulerConfig import SchedulerConfig
from ednaml.config.TransformationConfig import TransformationConfig
from ednaml.config.ModelConfig import ModelConfig
from ednaml.utils import merge_dictionary_on_key_with_copy

class EdnaMLConfig(BaseConfig):
    EXECUTION: ExecutionConfig
    DEPLOYMENT: DeploymentConfig
    SAVE: SaveConfig
    TRAIN_TRANSFORMATION: TransformationConfig
    TEST_TRANSFORMATION: TransformationConfig
    MODEL: ModelConfig
    LOSS: List[LossConfig]
    OPTIMIZER: List[
        OptimizerConfig
    ]  # one optimizer for each set of model params
    SCHEDULER: List[SchedulerConfig]  # one scheduler for each optimizer
    LOSS_OPTIMIZER: List[
        OptimizerConfig
    ]  # one optimizer for each loss with params
    LOSS_SCHEDULER: List[
        SchedulerConfig
    ]  # one scheduler for each loss_optimizer
    LOGGING: LoggingConfig

    extensions: List[str]

    def __init__(
        self, config_path: str, defaults: ConfigDefaults = ConfigDefaults()
    ):
        self.extensions = ["DEPLOYMENT"]
        ydict = self.read_path(config_path)

        self.EXECUTION = ExecutionConfig(ydict.get("EXECUTION", {}), defaults) #inside the exectution -- store execution object - -execution section
        self.DEPLOYMENT = DeploymentConfig(ydict.get("DEPLOYMENT", {}), defaults) #inside the exectution -- store execution object - -execution section
        self.SAVE = SaveConfig(ydict.get("SAVE", {}), defaults)
        self.TRAIN_TRANSFORMATION = TransformationConfig(
            dict(
                merge_dictionary_on_key_with_copy(ydict.get("TRANSFORMATION", {}), ydict.get("TRAIN_TRANSFORMATION", {}))
            ),
            defaults,
        )
        #print("SELF.TRAIN_TRANSFORMATION ::::::::::::::::::: ",self.TRAIN_TRANSFORMATION)
        self.TEST_TRANSFORMATION = TransformationConfig(
            dict(
                merge_dictionary_on_key_with_copy(ydict.get("TRANSFORMATION", {}), ydict.get("TEST_TRANSFORMATION", {}))
            ),
            defaults,
        )
        #print("SELF.TEST_TRANSFORMATION ::::::::::::::::::: ",self.TEST_TRANSFORMATION)

        self.MODEL = ModelConfig(
            ydict.get("MODEL", {}), defaults
        )  # No default MODEL itself, though it will be instantiated here? deal with this TODO
        self.LOSS = [
            LossConfig(loss_item, defaults)
            for loss_item in ydict.get("LOSS", [])
        ]  # No default LOSS itself --> it will be empty...

        # Default optimizer is Adam
        self.OPTIMIZER = [
            OptimizerConfig(optimizer_item, defaults)
            for optimizer_item in ydict.get("OPTIMIZER", [{}])
        ]
        self.SCHEDULER = [
            SchedulerConfig(scheduler_item, defaults)
            for scheduler_item in ydict.get("SCHEDULER", [{}])
        ]

        self.LOSS_OPTIMIZER = [
            OptimizerConfig(optimizer_item, defaults)
            for optimizer_item in ydict.get("LOSS_OPTIMIZER", [{}])
        ]
        self.LOSS_SCHEDULER = [
            SchedulerConfig(scheduler_item, defaults)
            for scheduler_item in ydict.get("LOSS_SCHEDULER", [{}])
        ]

        self.LOGGING = LoggingConfig(ydict.get("LOGGING", {}), defaults)

        # Things without defaults that MUST be provided: model ✅, train_dataloader, loss ✅, trainer TODO

    def export(self, mode="yaml"):
        dicts = json.dumps(self, default=config_serializer)  # or getvars()????
        dicta = json.loads(dicts)
        if mode == "yaml":
            return yaml.dump(dicta)
        elif mode == "json":
            return dicts
        elif mode == "dict":
            return dicta

    def extend(self, config_path: str, defaults: ConfigDefaults = ConfigDefaults()):
        """Extends the existing config object with fields from the provided config

        Args:
            config_path (str): The path to the config file to extend with
        """        
        
        ydict = self.read_path(config_path)
        for extension in self.extensions:
            if len(ydict.get(extension, {})) > 0:
                if extension == "DEPLOYMENT":
                    self.DEPLOYMENT = DeploymentConfig(ydict.get(extension, {}), defaults) 
                    return "Extended with DEPLOYMENT."
                else:
                    return "No valid extensions available."



    def read_path(self, path: os.PathLike) -> Dict[str,str]:
        """Reads the config at the path and yields a dictionary of the config

        Args:
            path (os.PathLike): Path to the config

        Raises:
            FileNotFoundError: Raised if path does not exist

        Returns:
            Dict[str,str]: Config as a python dictionary
        """        
        ydict = {} 

        if len(path) > 0:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "No file found for config at : %s" % path
                )
            else:
                with open(path, "r") as cfile:
                    ydict = yaml.safe_load(cfile.read().strip()) #loading the configuration file.
        return ydict

"""
Notes for LOSS_OPTIMIZER and LOSS_SCHEDULER
        Now we import the loss-optimizers and loss-scheduler
        Here, we need some additional code to make things work.
        
        Situation 1. LossBuilders, plus fully enumerated loss-optimizers. Simply load them into the config
        Situation 2. LossBuilders with no learnable params, and no loss-optimizers.
        Situation 3. LossBuilders with learnable parameters, and no loss-optimizers.
        Situation 4. LossBuilders, some with learnable parameters, and only corresponding loss-optimizers
        Situation 5. LossBuilders, some with learnable parameters, and less loss-optimizers
        Situation 6. LossBuilders, some with learnable parameters, and more loss-optimizers

        To cover all cases elegantly, we will simply provide:
        Situation 1. All loss-optimizers
        Situation 2. No way to check params here. Provide a default loss-optimizer
        Situation 3. No way to check params here. Provide a default loss-optimizer
        Situation 4. Provide enumerated loss-optimizers
        Situation 5. Provide enumerated loss-optimizers
        Situation 6. Provide enumerated loss-optimizers


        Then, EdnaML will:
        - construct a temporary dictionary of loss-optimizer configs, ordered by their name
        - For each loss-builder, check if there is a loss-optimizer with that name in the dict
        - If there is, then construct optimizer with those parameters
        - If there is not, then use the first optimizer's parameters to construct missing ones
        - this way, for each situation:
            Situation 1. No problems
            Situation 2. Default parameters used, but the optimizer builder will return None
            Situation 3. Default parameters used.
            Situation 4. No problems. The LossBuilders without params will return None
            Situation 5. Missing optimizers will be replaced with the first optimizer
            Situation 6. Extra optimizers will do nothing

        For Loss-schedulers, either there are enumerated schedulers, or a single default
        in EdnaML, we will match schedulers to the corresponding optimizers
        For optimizer without scheduler, create a scheduler using the first one
"""
