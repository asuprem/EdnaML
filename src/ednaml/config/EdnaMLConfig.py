# BaseConfig class to manage the input configurations. TODO
import os
import json
from turtle import update
from typing import Dict, List
import yaml
from ednaml.config import BaseConfig
from ednaml.config.DeploymentConfig import DeploymentConfig
from ednaml.config.ModelPluginConfig import ModelPluginConfig
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
from ednaml.config.StorageConfig import StorageConfig
from ednaml.utils import merge_dictionary_on_key_with_copy

class EdnaMLConfig(BaseConfig):
    EXECUTION: ExecutionConfig
    DEPLOYMENT: DeploymentConfig
    SAVE: SaveConfig
    TRAIN_TRANSFORMATION: TransformationConfig
    TEST_TRANSFORMATION: TransformationConfig
    MODEL: ModelConfig
    LOSS: List[LossConfig]
    MODEL_PLUGIN: Dict[str, ModelPluginConfig]
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
    STORAGE: StorageConfig

    extensions: List[str]

    def __init__(
        self, config_path: str, defaults: ConfigDefaults = ConfigDefaults()
    ):
        self.extensions = ["EXECUTION", "SAVE", "STORAGE", "TRANSFORMATION", "MODEL", "LOSS", "OPTIMIZER", "SCHEDULER", "LOSS_OPTIMIZER", "LOSS_SCHEDULER", "LOGGING", "DEPLOYMENT", "MODEL_PLUGIN"]  # TODO deal with other bits and pieces here!!!!!
        ydict = self.read_path(config_path)

        self._updateConfig(ydict, defaults, update_with_defaults=True)
    
    def _has_extension_verifier(self, ydict, extension, verification):
        return False if ydict.get(extension, verification) == verification else True

    def _updateConfig(self, ydict: Dict[str,str], defaults: ConfigDefaults, update_with_defaults: bool):
        added_extensions = []   # add control to check if extension was added or default was used...
        for extension in self.extensions:
            
            if extension == "EXECUTION":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.EXECUTION = ExecutionConfig(ydict.get(extension, {}), defaults) 
                    added_extensions.append([extension])
            elif extension == "DEPLOYMENT":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.DEPLOYMENT = DeploymentConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "STORAGE":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.STORAGE = StorageConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "SAVE":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.SAVE = SaveConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "TRANSFORMATION":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                has_train = self._has_extension_verifier(ydict, "TRAIN_TRANSFORMATION", {})
                if (has_extension and has_train) or update_with_defaults:
                    self.TRAIN_TRANSFORMATION = TransformationConfig(
                        dict(
                            merge_dictionary_on_key_with_copy(ydict.get(extension, {}), ydict.get("TRAIN_TRANSFORMATION", {}))
                        ),
                        defaults,
                    )
                    added_extensions.append(["TRAIN_TRANSFORMATION"])
                has_test = self._has_extension_verifier(ydict, "TEST_TRANSFORMATION", {})
                if (has_extension and has_test) or update_with_defaults:
                    self.TEST_TRANSFORMATION = TransformationConfig(
                        dict(
                            merge_dictionary_on_key_with_copy(ydict.get(extension, {}), ydict.get("TEST_TRANSFORMATION", {}))
                        ),
                        defaults,
                    )
                    added_extensions.append(["TEST_TRANSFORMATION"])
            elif extension == "MODEL":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.MODEL = ModelConfig(
                        ydict.get(extension, {}), defaults
                    )  # No default MODEL itself, though it will be instantiated here? deal with this TODO
                    added_extensions.append([extension])
            elif extension == "MODEL_PLUGIN":
                has_extension = self._has_extension_verifier(ydict, extension, [{}])
                if has_extension or update_with_defaults:
                    mp_list = [ModelPluginConfig(plugin_item, defaults) for plugin_item in ydict.get(extension, [{}])]
                    self.MODEL_PLUGIN = {item.PLUGIN_NAME: item for item in mp_list}
                    added_extensions.append([extension])
            elif extension == "LOSS":
                has_extension = self._has_extension_verifier(ydict, extension, [])
                if has_extension or update_with_defaults:
                    self.LOSS = [
                        LossConfig(loss_item, defaults)
                        for loss_item in ydict.get(extension, [])
                    ]  # No default LOSS itself --> it will be empty...
                    added_extensions.append([extension])
            elif extension == "OPTIMIZER":
                # Default optimizer is Adam
                has_extension = self._has_extension_verifier(ydict, extension, [{}])
                if has_extension or update_with_defaults:
                    self.OPTIMIZER = [
                        OptimizerConfig(optimizer_item, defaults)
                        for optimizer_item in ydict.get(extension, [{}])
                    ]
                    added_extensions.append([extension])
            elif extension == "SCHEDULER":
                has_extension = self._has_extension_verifier(ydict, extension, [{}])
                if has_extension or update_with_defaults:
                    self.SCHEDULER = [
                        SchedulerConfig(scheduler_item, defaults)
                        for scheduler_item in ydict.get(extension, [{}])
                    ]
                    added_extensions.append([extension])
            elif extension == "LOSS_OPTIMIZER":
                has_extension = self._has_extension_verifier(ydict, extension, [{}])
                if has_extension or update_with_defaults:
                    self.LOSS_OPTIMIZER = [
                        OptimizerConfig(optimizer_item, defaults)
                        for optimizer_item in ydict.get(extension, [{}])
                    ]
                    added_extensions.append([extension])
            elif extension == "LOSS_SCHEDULER":
                has_extension = self._has_extension_verifier(ydict, extension, [{}])
                if has_extension or update_with_defaults:
                    self.LOSS_SCHEDULER = [
                        SchedulerConfig(scheduler_item, defaults)
                        for scheduler_item in ydict.get(extension, [{}])
                    ]
                    added_extensions.append([extension])
            elif extension == "LOGGING":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.LOGGING = LoggingConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            else:
                raise RuntimeError("Somehow received unhandled extension %s"%str(extension))
        return added_extensions
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
        responses = {}
        if type(config_path) is str:
            config_path = [config_path]
        if type(config_path) is list:
            for cpath in config_path:
                responses[cpath] = self._extend(cpath, defaults)
        else:
            raise ValueError("Expected `list`, got %s"%type(config_path))

        return responses

    def _extend(self, config_path, defaults: ConfigDefaults):
        ydict = self.read_path(config_path)
        added_extensions = self._updateConfig(ydict, defaults, update_with_defaults=False)
        return "Extended with : %s"%", ".join([item[0] for item in added_extensions if len(item)>0])



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
