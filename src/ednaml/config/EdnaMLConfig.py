# BaseConfig class to manage the input configurations. TODO
import os
import json
from turtle import update
from typing import Any, Dict, List
import warnings
import yaml
from ednaml.config import BaseConfig
from ednaml.config.DeploymentConfig import DeploymentConfig
from ednaml.config.ExecutionDatareaderConfig import ExecutionDatareaderConfig
from ednaml.config.ModelPluginConfig import ModelPluginConfig
from ednaml.metrics.BaseMetric import BaseMetric
from ednaml.utils import config_serializer
from ednaml.config.ConfigDefaults import ConfigDefaults

from ednaml.config.ExecutionConfig import ExecutionConfig
from ednaml.config.LoggingConfig import LoggingConfig
from ednaml.config.LossConfig import LossConfig
from ednaml.config.OptimizerConfig import OptimizerConfig
from ednaml.config.MetricsConfig import MetricsConfig
from ednaml.config.SaveConfig import SaveConfig
from ednaml.config.SchedulerConfig import SchedulerConfig
from ednaml.config.TransformationConfig import TransformationConfig
from ednaml.config.ModelConfig import ModelConfig
from ednaml.config.StorageConfig import StorageConfig
from ednaml.utils import merge_dictionary_on_key_with_copy


class EdnaMLConfig(BaseConfig):
    EXECUTION: ExecutionConfig
    DATAREADER: ExecutionDatareaderConfig
    DEPLOYMENT: DeploymentConfig
    SAVE: SaveConfig
    TRAIN_TRANSFORMATION: TransformationConfig
    TEST_TRANSFORMATION: TransformationConfig
    MODEL: ModelConfig
    LOSS: List[LossConfig]
    MODEL_PLUGIN: Dict[str, ModelPluginConfig]
    OPTIMIZER: List[OptimizerConfig]  # one optimizer for each set of model params
    SCHEDULER: List[SchedulerConfig]  # one scheduler for each optimizer
    LOSS_OPTIMIZER: List[OptimizerConfig]  # one optimizer for each loss with params
    LOSS_SCHEDULER: List[SchedulerConfig]  # one scheduler for each loss_optimizer
    LOGGING: LoggingConfig
    STORAGE: Dict[str, StorageConfig]
    METRICS: MetricsConfig

    extensions: List[str]


    params: Dict[str, Any]
    metrics_enable: bool
    always_enable: bool
    step_enable: bool
    batch_enable: bool
    metrics: Dict[str, BaseMetric]
    always_metrics: List[str]
    immediate_metrics: List[str]
    step_metrics: List[str]
    batch_metrics: List[str]

    def __init__(
        self,
        config_path: List[str],
        defaults: ConfigDefaults = ConfigDefaults(),
        **kwargs
    ):
        self.params = {}
        self.extensions = [
            "EXECUTION",
            "DATAREADER",
            "SAVE",
            "STORAGE",
            "TRANSFORMATION",
            "MODEL",
            "LOSS",
            "OPTIMIZER",
            "SCHEDULER",
            "METRICS",
            "LOSS_OPTIMIZER",
            "LOSS_SCHEDULER",
            "LOGGING",
            "DEPLOYMENT",
            "MODEL_PLUGIN",
        ]  # TODO deal with other bits and pieces here!!!!!
        ydict = self.merge(
            [self.read_path(config_component) for config_component in config_path]
        )
        config_inject = kwargs.get("config_inject", None)
        if config_inject is not None and type(config_inject) is list:
            self.config_inject(ydict, config_inject)
        self._updateConfig(ydict, defaults, update_with_defaults=True)

        self.metrics_enable = False
        self.always_enable = False
        self.step_enable = False
        self.batch_enable = False

        self.always_metrics = []
        self.immediate_metrics = []
        self.step_metrics = []
        self.batch_metrics = []

        self.metrics = {}

    def config_inject(self, ydict, config_inject: List[List[str]]):
        for inject in config_inject:
            inject_key = inject[0]
            inject_value = inject[1]
            if len(inject) == 3:
                inject_kwarg = inject[2]
            else:
                inject_kwarg = None
            try:
                self._setinject(
                    ydict, inject_key.split("."), inject_value, inject_kwarg
                )
                print(
                    "Injected key-value pair:  {key}, {value}".format(
                        key=inject_key, value=inject_value
                    )
                )
            except KeyError:
                pass

    def _setinject(self, d, inject_key, inject_val, inject_kwarg=None):
        for elem in inject_key[:-1]:
            d = d[elem]
        if inject_kwarg is not None:
            d[inject_key[-1]][inject_kwarg] = inject_val
        else:
            d[inject_key[-1]] = inject_val

    def _has_extension_verifier(self, ydict, extension, verification):
        return False if ydict.get(extension, verification) == verification else True

    def _updateConfig(
        self,
        ydict: Dict[str, str],
        defaults: ConfigDefaults,
        update_with_defaults: bool,
    ):
        added_extensions = (
            []
        )  # add control to check if extension was added or default was used...
        for extension in self.extensions:

            if extension == "EXECUTION":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.EXECUTION = ExecutionConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "DEPLOYMENT":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.DEPLOYMENT = DeploymentConfig(
                        ydict.get(extension, {}), defaults
                    )
                    added_extensions.append([extension])
            elif extension == "STORAGE":
                has_extension = self._has_extension_verifier(ydict, extension, [])
                if has_extension or update_with_defaults:
                    storage_list: List[StorageConfig] = [
                        StorageConfig(storage_item, defaults)
                        for storage_item in ydict.get(extension, [])
                    ]
                    self.STORAGE = {item.STORAGE_NAME: item for item in storage_list}
                    added_extensions.append([extension])
            elif extension == "SAVE":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.SAVE = SaveConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "METRICS":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.METRICS = MetricsConfig(ydict.get(extension, {}), defaults)
                    added_extensions.append([extension])
            elif extension == "DATAREADER":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                if has_extension or update_with_defaults:
                    self.DATAREADER = ExecutionDatareaderConfig(
                        ydict.get(extension, {}), defaults
                    )
                    added_extensions.append([extension])
            elif extension == "TRANSFORMATION":
                has_extension = self._has_extension_verifier(ydict, extension, {})
                has_train = self._has_extension_verifier(
                    ydict, "TRAIN_TRANSFORMATION", {}
                )
                if (has_extension and has_train) or update_with_defaults:
                    self.TRAIN_TRANSFORMATION = TransformationConfig(
                        dict(
                            merge_dictionary_on_key_with_copy(
                                ydict.get(extension, {}),
                                ydict.get("TRAIN_TRANSFORMATION", {}),
                            )
                        ),
                        defaults,
                    )
                    added_extensions.append(["TRAIN_TRANSFORMATION"])
                has_test = self._has_extension_verifier(
                    ydict, "TEST_TRANSFORMATION", {}
                )
                if (has_extension and has_test) or update_with_defaults:
                    self.TEST_TRANSFORMATION = TransformationConfig(
                        dict(
                            merge_dictionary_on_key_with_copy(
                                ydict.get(extension, {}),
                                ydict.get("TEST_TRANSFORMATION", {}),
                            )
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
                has_extension = self._has_extension_verifier(ydict, extension, [])
                if has_extension or update_with_defaults:
                    mp_list: List[ModelPluginConfig] = [
                        ModelPluginConfig(plugin_item, defaults)
                        for plugin_item in ydict.get(extension, [])
                    ]
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
                raise RuntimeError(
                    "Somehow received unhandled extension %s" % str(extension)
                )
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

    def extend(
        self, config_path: str, defaults: ConfigDefaults = ConfigDefaults(), **kwargs
    ):
        """Extends the existing config object with fields from the provided config

        Args:
            config_path (str): The path to the config file to extend with
        """
        responses = {}
        if type(config_path) is str:
            config_path = [config_path]
        if type(config_path) is list:
            for cpath in config_path:
                responses[cpath] = self._extend(cpath, defaults, **kwargs)
        else:
            raise ValueError("Expected `list`, got %s" % type(config_path))

        return responses

    def _extend(self, config_path, defaults: ConfigDefaults, **kwargs):
        ydict = self.read_path(config_path)
        config_inject = kwargs.get("config_inject", None)
        if config_inject is not None and type(config_inject) is list:
            self.config_inject(ydict, config_inject)
        added_extensions = self._updateConfig(
            ydict, defaults, update_with_defaults=False
        )
        return "Extended with : %s" % ", ".join(
            [item[0] for item in added_extensions if len(item) > 0]
        )

    def read_path(self, path: os.PathLike) -> Dict[str, str]:
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
                raise FileNotFoundError("No file found for config at : %s" % path)
            else:
                with open(path, "r") as cfile:
                    ydict = yaml.safe_load(
                        cfile.read().strip()
                    )  # loading the configuration file.
        return ydict

    def _merge(
        self,
        base_config: Dict[str, str],
        extension: Dict[str, str],
        recursion_flag: bool = False,
    ):
        """Combines two config files
        If capital lettered key is present it'll be replaced
        If small lettered key is present, it'll be deleted and considered the key in second object only

        Args:
            first object: Dict (base config)
            second object: Dict (second config)
        Returns:
            Dict[str,str]: Config as a combined python dictionary
        """
        # if(recursion_flag):
        #    to_keep = lambda key: key.islower()
        #    {k: v for k, v in base_config.items() if k not in keyfilter(to_keep, base_config)}
        lower_flag = False
        upper_flag = False
        for extension_key in extension.keys():
            if extension_key in base_config:
                if (
                    extension_key.islower()
                ):  # If NOT a built-in key, we will do a full replacement of the parent...
                    lower_flag = True
                else:  # If it is a built-in key
                    upper_flag = True
                    if isinstance(base_config[extension_key], dict):  # If it is a dict
                        if isinstance(extension[extension_key], dict):
                            # Recurse if key is a Dictionary with elements inside.
                            base_config[extension_key] = self._merge(
                                base_config[extension_key],
                                extension[extension_key],
                                recursion_flag=True,
                            )
                        else:
                            base_config[extension_key] = extension[extension_key]
                            warnings.warn(
                                "Key {key}: Dict key in base configuration is not a dictionary in replacement extension. \n\tBase: {base_str}\n\tExtension: {ext_str}".format(
                                    key=extension_key,
                                    base_str=str(base_config(extension_key)),
                                    ext_str=str(base_config(extension_key)),
                                )
                            )
                    else:  # The base config key is not a dictionary...we just do replacement. Don't care if replacement is dictionary
                        base_config[extension_key] = extension[extension_key]
            else:  # a key in extensionis NOT in base_config, so we don't really care...
                base_config[extension_key] = extension[extension_key]
        if lower_flag and upper_flag:
            raise ValueError("built-in and custom keys cannot mix")
        if lower_flag:
            return extension
        return base_config

    def merge(self, config_paths: List[str]):
        """Combines multiple config files
        Args:
            first object: Dict array (base config as index 0)
        Returns:
            Dict[str,str]: Config as a combined python dictionary
        """
        base_config = config_paths[0]
        for extension_config in config_paths[1:]:
            base_config = self._merge(base_config, extension_config)
        return base_config

    def save(self, path: os.PathLike):
        with open(path, "w") as write_file:
            write_file.write(self.export())


    def addMetrics(self, metrics_list: List[BaseMetric], epoch, step):
        self.immediate_metrics: List[BaseMetric] = []
        if len(metrics_list) > 0:
            # Check for metrics that are executed only once, e.g. now
            for metric in metrics_list:
                self.metrics[metric.metric_name] = metric
                if metric.metric_trigger == "once":
                    # Metric is triggered only once.
                    # We will trigger it at the end of addMetrics()

                    self.immediate_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "always":
                    # Metric is `always` triggered. Always has different meanings for each Manager. We will add it to our bookkeeping
                    self.always_metrics.append(metric.metric_name)

                elif metric.metric_trigger == "step":
                    self.step_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "batch":
                    self.batch_metrics.append(metric.metric_name)
                else:
                    raise ValueError("metric_trigger %s is not supported"%metric.metric_trigger)
            self.metrics_enable = True
        else:
            self.metrics_enable = False

        if self.metrics_enable:
            for metric_name in self.immediate_metrics:
                self.metrics[metric_name].update(epoch, step, params=self.params)

            if len(self.always_metrics):
                self.always_enable = True # Enable always metrics
            if len(self.step_metrics):
                self.step_enable = True # Enable step metrics
            if len(self.batch_metrics):
                self.batch_enable = True # Enable batch metrics

    # For now, we ignore always metrics
    def updateStepMetrics(self, epoch, step):
        for metric_name in self.step_metrics:
            self.metrics[metric_name].update(epoch=epoch, step=step, params=self.params)

    def updateBatchMetrics(self, epoch, step):
        for metric_name in self.batch_metrics:
            self.metrics[metric_name].update(epoch=epoch, step=step, params=self.params)




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
