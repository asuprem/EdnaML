# BaseConfig class to manage the input configurations. TODO

import json
from typing import List
import yaml
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


class EdnaMLConfig:
    EXECUTION: ExecutionConfig
    SAVE: SaveConfig
    TRANSFORMATION: TransformationConfig
    MODEL: ModelConfig
    LOSS: List[LossConfig]
    OPTIMIZER: List[OptimizerConfig]  # one optimizer for each set of model params
    SCHEDULER: List[SchedulerConfig]  # one scheduler for each optimizer
    LOSS_OPTIMIZER: List[OptimizerConfig]  # one optimizer for each loss with params
    LOSS_SCHEDULER: List[SchedulerConfig]  # one scheduler for each loss_optimizer
    LOGGING: LoggingConfig

    def __init__(self, config_path: str, defaults: ConfigDefaults = ConfigDefaults()):
        ydict = {}
        with open(config_path, "r") as cfile:
            ydict = yaml.safe_load(cfile.read().strip())

        self.EXECUTION = ExecutionConfig(ydict.get("EXECUTION", {}), defaults)
        self.SAVE = SaveConfig(ydict.get("SAVE", {}), defaults)
        self.TRANSFORMATION = TransformationConfig(
            ydict.get("TRANSFORMATION", {}), defaults
        )
        self.MODEL = ModelConfig(
            ydict.get("MODEL", {}), defaults
        )  # No default MODEL itself, though it will be instantiated here? deal with this TODO
        self.LOSS = [
            LossConfig(loss_item, defaults) for loss_item in ydict.get("LOSS", [])
        ]  # No default LOSS itself

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
        dicts = json.dumps(self, default=config_serializer)
        dicta = json.loads(dicts)
        if mode == "yaml":
            return yaml.dump(dicta)
        elif mode == "json":
            return dicts
        elif mode == "dict":
            return dicta


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
