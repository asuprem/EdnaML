import importlib
import os, shutil, logging, glob, re
from types import FunctionType, MethodType
from typing import Callable, Dict, List, Type, Union
import warnings
from torchinfo import ModelStatistics
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.config.LossConfig import LossConfig
from ednaml.config.ModelConfig import ModelConfig
from ednaml.core import EdnaMLBase
from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators import Generator
from ednaml.loss.builders import LossBuilder
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.optimizer import BaseOptimizer
from ednaml.optimizer.StandardLossOptimizer import StandardLossOptimizer
from ednaml.loss.builders import ClassificationLossBuilder
from ednaml.plugins.ModelPlugin import ModelPlugin
from ednaml.trainer.BaseTrainer import BaseTrainer
from ednaml.utils import locate_class, path_import
import ednaml.utils
import torch
from torchinfo import summary
from ednaml.utils.LabelMetadata import LabelMetadata
import ednaml.utils.web
from ednaml.utils.SaveMetadata import SaveMetadata
import logging
import ednaml.core.decorators

"""TODO

Verbosity = 0 -> create no logger and use a dummy that print nothing anywhere
"""


class EdnaML(EdnaMLBase):
    labelMetadata: LabelMetadata
    modelStatistics: ModelStatistics
    model: ModelAbstract = None
    loss_function_array: List[LossBuilder]
    loss_optimizer_array: List[torch.optim.Optimizer]
    optimizer: List[torch.optim.Optimizer]
    scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    previous_stop: int
    trainer: BaseTrainer
    crawler: Crawler
    train_generator: Generator
    test_generator: Generator
    cfg: EdnaMLConfig
    decorator_reference: Dict[str,Type[MethodType]]
    plugins: List[ModelPlugin] = []

    def __init__(
        self,
        config: Union[List[str], str] = "",
        mode: str = "train",
        weights: str = None,
        logger: logging.Logger = None,
        verbose: int = 2,
        **kwargs
    ):
        """Initializes the EdnaML object with the associated config, mode, weights, and verbosity.
        Sets up the logger, as well as local logger save directory.

        Args:
            config (str, optional): Path to the Edna config file. Defaults to "config.yaml".
            mode (str, optional): Either `train` or `test`. Defaults to "train".
            weights (str, optional): The path to the weights file. Defaults to None.
            logger (logging.Logger, optional): A logger. If no logger is provided, EdnaML will construct its own. Defaults to None.
            verbose (int, optional): Logging verbosity. Defaults to 2.

        Kwargs:
            load_epoch: If you want EdnaML to load a model from a specific saved epoch. The path construction will be inferred from the config file
            add_filehandler: If you want logger to write to log file
            add_streamhandler: If you want logger to write to stdout
            logger_save_name: If you want logger name to be something other than default constructed name from config
            test_only: Under this condition, if in test mode, only model and dataloaders will be created. Optimizers, schedulers will be empty.

        """

        self.config = config
        self.mode = mode
        self.weights = weights
        self.pretrained_weights = None
        self.verbose = verbose
        self.gpus = torch.cuda.device_count()

        # Added configuration extentions
        if type(self.config) is list:
            self.cfg = EdnaMLConfig(config[0])
            for cfg_item in config[1:]:
                msg = self.cfg.extend(cfg_item)
                self.logger.info(str(msg))
        else:
            self.cfg = EdnaMLConfig(config) 
        
        
        self.saveMetadata = SaveMetadata(
            self.cfg, **kwargs
        )  # <-- for changing the logger name...
        os.makedirs(self.saveMetadata.MODEL_SAVE_FOLDER, exist_ok=True)

        self.logger = self.buildLogger(logger=logger, **kwargs)
        self.previous_stop = -1
        self.load_epoch = kwargs.get("load_epoch", None)
        self.test_only = kwargs.get("test_only", False)
        if self.test_only and self.mode == "train":
            raise ValueError(
                "Cannot have `test_only` and be in training mode. Switch to"
                " `test` mode."
            )
        self.resetQueueArray:List[MethodType] = [self.resetCrawlerQueues, self.resetGeneratorQueues, self.resetModelQueues, self.resetOptimizerQueues, self.resetLossBuilderQueue, self.resetTrainerQueue]
        self.resetQueueArray += self.addResetQueues()
        self.resetQueues()

        self.decorator_reference = {
            "crawler": self.addCrawlerClass,
            "model": self.addModelClass,
            "trainer": self.addTrainerClass,
            #"storage": self.addStorageClass,
            "generator": self.addGeneratorClass,
            "model_plugin": self.addPlugins,
        }

    def addResetQueues(self):
        return []
    def resetCrawlerQueues(self):
        self._crawlerClassQueue = None
        self._crawlerArgsQueue = None
        self._crawlerArgsQueueFlag = False
        self._crawlerClassQueueFlag = False
        self._crawlerInstanceQueue = None
        self._crawlerInstanceQueueFlag = False
    def resetGeneratorQueues(self):
        self._generatorClassQueue = None
        self._generatorArgsQueue = None
        self._generatorArgsQueueFlag = False
        self._generatorClassQueueFlag = False
        self._trainGeneratorInstanceQueue = None
        self._trainGeneratorInstanceQueueFlag = False
        self._testGeneratorInstanceQueue = None
        self._testGeneratorInstanceQueueFlag = False
    def resetModelQueues(self):
        self._modelBuilderQueue = None
        self._modelBuilderQueueFlag = False
        self._modelConfigQueue = None
        self._modelConfigQueueFlag = False
        self._modelQueue = None
        self._modelQueueFlag = False
        self._modelClassQueue = None
        self._modelClassQueueFlag = False
        self._modelArgsQueue = None
        self._modelArgsQueueFlag = False
    def resetOptimizerQueues(self):
        self._optimizerQueue = []
        self._optimizerNameQueue = []
        self._optimizerQueueFlag = False
    def resetLossBuilderQueue(self):
        self._lossBuilderQueue = []
        self._lossBuilderQueueFlag = False
    def resetTrainerQueue(self):
        self._trainerClassQueue = None
        self._trainerClassQueueFlag = False
        self._trainerInstanceQueue = None
        self._trainerInstanceQueueFlag = False

    def resetQueues(self):
        """Resets the `apply()` queue"""
        for queue_function in self.resetQueueArray:
            queue_function()

    def recordVars(self):
        """recordVars() completes initial setup, allowing you to proceed with the core ml pipeline
        of dataloading, model building, etc
        """

        self.epochs = self.cfg.EXECUTION.EPOCHS
        self.skipeval = self.cfg.EXECUTION.SKIPEVAL
        self.step_verbose = self.cfg.LOGGING.STEP_VERBOSE
        self.save_frequency = self.cfg.SAVE.SAVE_FREQUENCY
        self.test_frequency = self.cfg.EXECUTION.TEST_FREQUENCY
        self.fp16 = self.cfg.EXECUTION.FP16

    def downloadModelWeights(self):
        """Downloads model weights specified in the configuration if `weights` were not passed into EdnaML and if model weights are supported.

        TODO -- do not throw error for no weights or missing base, if this is a new architecture to be trained from scratch
        Raises:
            Warning: If there are no pre-downloaded weights, and the model architecture is unsupported
        """
        if self.weights is not None:
            self.logger.info(
                "Not downloading weights. Weights path already provided."
            )

        if self.mode == "train":
            self._download_weights_from_base(self.cfg.MODEL.MODEL_BASE)
        else:
            if self.weights is None:
                warnings.warn(
                    "Mode is `test` but weights is `None`. This will cause"
                    " issues when EdnaML attempts to load weights"
                )

    def _download_weights_from_base(self, model_base: str):
        """Downloads weights from a model_base parameter directly from pytorch's servers.

        Args:
            model_base (str): A supported `model_base`, e.g. resnet18, resnet34. See `utils.web.model_weights`.
        """
        from ednaml.utils import model_weights

        if model_base in model_weights:
            if os.path.exists(model_weights[model_base][1]):
                pass
            else:
                self.logger.info(
                    "Model weights file {} does not exist. Downloading.".format(
                        model_weights[model_base][1]
                    )
                )
                ednaml.utils.web.download(
                    model_weights[model_base][1], model_weights[model_base][0]
                )
            self.pretrained_weights = model_weights[model_base][1]
        else:
            warnings.warn(
                "Model %s is not available. Please choose one of the following:"
                " %s if you want to load pretrained weights"
                % (model_base, str(model_weights.keys()))
            )

    def apply(self, **kwargs):
        """Applies the internal configuration for EdnaML"""
        self.printConfiguration()
        self.downloadModelWeights()
        self.setPreviousStop()

        self.buildDataloaders()

        self.buildModel()
        self.loadWeights()
        self.getModelSummary(**kwargs) 
        self.buildOptimizer()
        self.buildScheduler()

        self.buildLossArray()
        self.buildLossOptimizer()
        self.buildLossScheduler()

        self.buildTrainer()

        self.resetQueues()

    def train(self):
        self.trainer.train(continue_epoch=self.previous_stop + 1)  #

    def eval(self):
        return self.trainer.evaluate()

    def addTrainerClass(self, trainerClass: Type[BaseTrainer]):
        self._trainerClassQueue = trainerClass
        self._trainerClassQueueFlag = True

    def addTrainer(self, trainer: BaseTrainer):
        self._trainerInstanceQueue = trainer
        self._trainerInstanceQueueFlag = True

    def buildTrainer(self):
        """Builds the EdnaML trainer and sets it up"""
        if self._trainerClassQueueFlag:
            ExecutionTrainer = self._trainerClassQueue
        else:
            ExecutionTrainer: Type[BaseTrainer] = locate_class(
                subpackage="trainer", classpackage=self.cfg.EXECUTION.TRAINER
            )
            self.logger.info(
                "Loaded {} from {} to build Trainer".format(
                    self.cfg.EXECUTION.TRAINER, "ednaml.trainer"
                )
            )

        if self._trainerInstanceQueueFlag:
            self.trainer = self._trainerInstanceQueue
        else:
            self.trainer = ExecutionTrainer(
                model=self.model,
                loss_fn=self.loss_function_array,
                optimizer=self.optimizer,
                loss_optimizer=self.loss_optimizer_array,
                scheduler=self.scheduler,
                loss_scheduler=self.loss_scheduler,
                train_loader=self.train_generator.dataloader,
                test_loader=self.test_generator.dataloader,
                epochs=self.cfg.EXECUTION.EPOCHS,
                skipeval=self.cfg.EXECUTION.SKIPEVAL,
                logger=self.logger,
                crawler=self.crawler,
                config=self.cfg,
                labels=self.labelMetadata,
                **self.cfg.EXECUTION.TRAINER_ARGS
            )
            self.trainer.apply(
                step_verbose=self.cfg.LOGGING.STEP_VERBOSE,
                save_frequency=self.cfg.SAVE.SAVE_FREQUENCY,
                test_frequency=self.cfg.EXECUTION.TEST_FREQUENCY,
                save_directory=self.saveMetadata.MODEL_SAVE_FOLDER,
                save_backup=self.cfg.SAVE.DRIVE_BACKUP,
                backup_directory=self.saveMetadata.CHECKPOINT_DIRECTORY,
                gpus=self.gpus,
                fp16=self.cfg.EXECUTION.FP16,
                model_save_name=self.saveMetadata.MODEL_SAVE_NAME,
                logger_file=self.saveMetadata.LOGGER_SAVE_NAME,
            )

    def setPreviousStop(self):
        """Sets the previous stop"""
        self.previous_stop = self.getPreviousStop()

    def getPreviousStop(self) -> int:
        """Gets the previous stop, if any, of the trainable model by checking local save directory, as well as a network directory."""
        if self.cfg.SAVE.DRIVE_BACKUP:
            fl_list = glob.glob(
                os.path.join(self.saveMetadata.CHECKPOINT_DIRECTORY, "*.pth")
            )
        else:
            fl_list = glob.glob(
                os.path.join(self.saveMetadata.MODEL_SAVE_FOLDER, "*.pth")
            )
        _re = re.compile(r".*epoch([0-9]+)\.pth")
        previous_stop = [
            int(item[1])
            for item in [_re.search(item) for item in fl_list]
            if item is not None
        ]
        if len(previous_stop) == 0:
            self.logger.info(
                "No previous stop detected. Will start from epoch 0"
            )
            return -1
        else:
            self.logger.info(
                "Previous stop detected. Will attempt to resume from epoch %i"
                % max(previous_stop)
            )
            return max(previous_stop)

    def addOptimizer(
        self, optimizer: torch.optim.Optimizer, parameter_group="opt-1"
    ):
        self._optimizerQueue.append(optimizer)
        self._optimizerNameQueue.append(parameter_group)
        self._optimizerQueueFlag = True

    def buildOptimizer(self):
        """Builds the optimizer for the model"""
        if self.test_only:
            self.optimizer = []
            self.logger.info(
                "Skipping optimizer building step in `test_only` mode"
            )
        else:
            if self._optimizerQueueFlag:
                self.optimizer = [item for item in self._optimizerQueue]
            else:
                optimizer_builder: Type[BaseOptimizer] = locate_class(
                    subpackage="optimizer",
                    classpackage=self.cfg.EXECUTION.OPTIMIZER_BUILDER,
                )
                self.logger.info(
                    "Loaded {} from {} to build Optimizer model".format(
                        self.cfg.EXECUTION.OPTIMIZER_BUILDER, "ednaml.optimizer"
                    )
                )

                # Optimizers are in a list...
                OPT_array = [
                    optimizer_builder(
                        name=optimizer_item.OPTIMIZER_NAME,
                        optimizer=optimizer_item.OPTIMIZER,
                        base_lr=optimizer_item.BASE_LR,
                        lr_bias=optimizer_item.LR_BIAS_FACTOR,
                        weight_decay=optimizer_item.WEIGHT_DECAY,
                        weight_bias=optimizer_item.WEIGHT_BIAS_FACTOR,
                        opt_kwargs=optimizer_item.OPTIMIZER_KWARGS,
                        gpus=self.gpus,
                    )
                    for optimizer_item in self.cfg.OPTIMIZER
                ]
                self.optimizer = [
                    OPT.build(
                        self.model.getParameterGroup(
                            self.cfg.OPTIMIZER[idx].OPTIMIZER_NAME
                        )
                    )
                    for idx, OPT in enumerate(OPT_array)
                ]  # TODO make this a dictionary???
            self.logger.info("Built optimizer")

    def buildScheduler(self):
        """Builds the scheduler for the model"""
        if self.test_only:
            self.scheduler = []
            self.logger.info(
                "Skipping scheduler building step in `test_only` mode"
            )
        else:
            self.scheduler = [None] * len(self.cfg.SCHEDULER)
            for idx, scheduler_item in enumerate(self.cfg.SCHEDULER):
                try:  # We first check if scheduler is part of torch's provided schedulers.
                    scheduler = locate_class(
                        package="torch.optim",
                        subpackage="lr_scheduler",
                        classpackage=scheduler_item.LR_SCHEDULER,
                    )
                except (
                    ModuleNotFoundError,
                    AttributeError,
                ):  # If it fails, then we try to import from schedulers implemented in scheduler/ folder
                    scheduler = locate_class(
                        subpackage="scheduler",
                        classpackage=scheduler_item.LR_SCHEDULER,
                    )
                self.scheduler[idx] = scheduler(
                    self.optimizer[idx],
                    last_epoch=-1,
                    **scheduler_item.LR_KWARGS
                )
            self.logger.info("Built scheduler")



    def addLossBuilder(
        self, loss_list, loss_lambdas, loss_kwargs, loss_name, loss_label
    ):
        self._lossBuilderQueue.append(
            LossConfig(
                {
                    "LOSSES": loss_list,
                    "LAMBDAS": loss_lambdas,
                    "KWARGS": loss_kwargs,
                    "LABEL": loss_label,
                    "NAME": loss_name,
                }
            )
        )
        self._lossBuilderQueueFlag = True

    def buildLossArray(self):
        """Builds the loss function array using the LOSS list in the provided configuration"""
        if self.test_only:
            self.loss_function_array = []
            self.logger.info(
                "Skipping loss function array building step in `test_only` mode"
            )
        else:
            self.loss_function_array = [
                ClassificationLossBuilder(
                    loss_functions=loss_item.LOSSES,
                    loss_lambda=loss_item.LAMBDAS,
                    loss_kwargs=loss_item.KWARGS,
                    name=loss_item.NAME,  # get("NAME", None),
                    label=loss_item.LABEL,  # get("LABEL", None),
                    metadata=self.labelMetadata,
                    **{"logger": self.logger}
                )
                for loss_item in self.cfg.LOSS
            ]
            if self._lossBuilderQueueFlag:
                self.loss_function_array += [
                    ClassificationLossBuilder(
                        loss_functions=loss_item.LOSSES,
                        loss_lambda=loss_item.LAMBDAS,
                        loss_kwargs=loss_item.KWARGS,
                        name=loss_item.NAME,  # get("NAME", None),
                        label=loss_item.LABEL,  # get("LABEL", None),
                        metadata=self.labelMetadata,
                        **{"logger": self.logger}
                    )
                    for loss_item in self._lossBuilderQueue
                ]
            self.logger.info("Built loss function")

    def buildLossOptimizer(self):
        """Builds the Optimizer for loss functions, if the loss functions have learnable parameters (e.g. proxyNCA loss)

        self.loss_function_array contains a list of LossBuilders. Each LossBuilder
        is for a specific output. Here, we build an array of StandardLossOptimizers,
        one StandardLossOptimizer for each LossBuilder. Each StandardLossOptimizer
        takes as input the same arguments as an Optimizer. However, the name parameter
        should be the name of the LossBuilder it is targeting.

        If there is no LOSS_OPTIMIZER section in the configuration, the EdnaML config creates a default
        LOSS_OPTIMIZER, whose parameters we will use.

        """
        if self.test_only:
            self.loss_optimizer_array = []
            self.logger.info(
                "Skipping loss-scheduler building step in `test_only` mode"
            )
        else:
            loss_optimizer_name_dict = {
                loss_optim_item.OPTIMIZER_NAME: loss_optim_item
                for loss_optim_item in self.cfg.LOSS_OPTIMIZER
            }
            initial_key = list(loss_optimizer_name_dict.keys())[0]
            LOSS_OPT: List[StandardLossOptimizer] = [None] * len(
                self.loss_function_array
            )
            for idx, loss_builder in enumerate(self.loss_function_array):
                if loss_builder.loss_labelname in loss_optimizer_name_dict:
                    # Means we have an optimizer corresponding to this loss
                    lookup_key = loss_builder.loss_labelname
                else:  # We will use the first optimizer (either default or otherwise, etc, for this)
                    lookup_key = initial_key
                LOSS_OPT[idx] = StandardLossOptimizer(
                    name=loss_optimizer_name_dict[lookup_key].OPTIMIZER_NAME,
                    optimizer=loss_optimizer_name_dict[lookup_key].OPTIMIZER,
                    base_lr=loss_optimizer_name_dict[lookup_key].BASE_LR,
                    lr_bias=loss_optimizer_name_dict[lookup_key].LR_BIAS_FACTOR,
                    gpus=self.gpus,
                    weight_bias=loss_optimizer_name_dict[
                        lookup_key
                    ].WEIGHT_BIAS_FACTOR,
                    weight_decay=loss_optimizer_name_dict[
                        lookup_key
                    ].WEIGHT_DECAY,
                    opt_kwargs=loss_optimizer_name_dict[
                        lookup_key
                    ].OPTIMIZER_KWARGS,
                )

            # Note: build returns None if there are no differentiable parameters
            self.loss_optimizer_array = [
                loss_opt.build(loss_builder=self.loss_function_array[idx])
                for idx, loss_opt in enumerate(LOSS_OPT)
            ]
            self.logger.info("Built loss optimizer")

    def buildLossScheduler(self):
        """Builds the scheduler for the loss functions, if the functions have learnable parameters and corresponding optimizer."""
        if self.test_only:
            self.loss_scheduler = []
            self.logger.info(
                "Skipping loss-scheduler building step in `test_only` mode"
            )
        else:
            loss_scheduler_name_dict = {
                loss_schedule_item.SCHEDULER_NAME: loss_schedule_item
                for loss_schedule_item in self.cfg.LOSS_SCHEDULER
            }
            initial_key = list(loss_scheduler_name_dict.keys())[0]
            self.loss_scheduler = [None] * len(self.loss_optimizer_array)

            for idx, loss_optimizer in enumerate(self.loss_optimizer_array):
                if (
                    loss_optimizer is not None
                ):  # In case loss has differentiable parameters, so the optimizer is not None...we look for the loss name
                    if (
                        self.loss_function_array[idx].loss_labelname
                        in loss_scheduler_name_dict
                    ):
                        lookup_key = self.loss_function_array[
                            idx
                        ].loss_labelname
                    else:
                        lookup_key = initial_key

                    try:  # We first check if scheduler is part of torch's provided schedulers.
                        loss_scheduler = importlib.import_module(
                            loss_scheduler_name_dict[lookup_key].LR_SCHEDULER,
                            package="torch.optim.lr_scheduler",
                        )
                    except (
                        ModuleNotFoundError,
                        AttributeError,
                    ):  # If it fails, then we try to import from schedulers implemented in scheduler/ folder
                        loss_scheduler = importlib.import_module(
                            loss_scheduler_name_dict[lookup_key].LR_SCHEDULER,
                            package="ednaml.scheduler",
                        )
                    self.loss_scheduler[idx] = loss_scheduler(
                        loss_optimizer,
                        last_epoch=-1,
                        **loss_scheduler_name_dict[lookup_key].LR_KWARGS
                    )
                self.logger.info("Built loss scheduler")

    def _setModelTestMode(self):
        """Sets model to test mode if EdnaML is in testing mode"""
        if self.mode == "test":
            self.model.eval()

    def _setModelTrainMode(self):
        """Sets the model to train mode if EdnaML is in training mode"""
        if self.mode == "train":
            self.model.train()

    def _covert_model_kwargs(self) -> Dict[str, int]:
        """Converts the model_kwargs inside config into the correct format, depending on whether it is provided directly in yaml format, or as a json string

        Returns:
            Dict[str,Union[str,int]]: Corrected model_kwargs dictionary
        """

        if (
            type(self.cfg.MODEL.MODEL_KWARGS) is dict
        ):  # Compatibility with old configs. TODO fix all old configs.
            model_kwargs_dict = self.cfg.MODEL.MODEL_KWARGS
        elif type(self.cfg.MODEL.MODEL_KWARGS) is None:
            model_kwargs_dict = {}
        elif type(self.cfg.MODEL.MODEL_KWARGS) is str:
            raise ValueError("Outdated model_kwargs as str")
            # model_kwargs_dict = json.loads(self.cfg.MODEL.MODEL_KWARGS)
        return model_kwargs_dict

    def buildModel(self):
        """Builds an EdnaML model using the configuration. If there are pretrained weights, they are provided through the config to initialize the model."""
        if self._modelBuilderQueueFlag:
            model_builder = self._modelBuilderQueue
        else:
            model_builder = locate_class(
                subpackage="models", classpackage=self.cfg.MODEL.BUILDER
            )
        self.logger.info(
            "Loaded {} from {} to build model".format(
                self.cfg.MODEL.BUILDER, "ednaml.models"
            )
        )

        if self._modelConfigQueueFlag:
            self.cfg.MODEL = self._modelConfigQueue

        # model_kwargs = self._covert_model_kwargs()

        # TODO!!!!!!!
        if self._modelQueueFlag:
            self.model = self._modelQueue
        else:
            if self._modelArgsQueueFlag:
                self.cfg.MODEL.MODEL_KWARGS = self._modelArgsQueue

            if self._modelClassQueueFlag:
                self.cfg.MODEL.MODEL_ARCH = self._modelClassQueue.__name__
                arch = self._modelClassQueue
            else:
                arch = self.cfg.MODEL.MODEL_ARCH
            self.model: ModelAbstract = model_builder(
                arch=arch,
                base=self.cfg.MODEL.MODEL_BASE,
                weights=self.pretrained_weights,
                metadata=self.labelMetadata,
                normalization=self.cfg.MODEL.MODEL_NORMALIZATION,
                parameter_groups=self.cfg.MODEL.PARAMETER_GROUPS,
                **self.cfg.MODEL.MODEL_KWARGS
            )
        self.logger.info(
            "Finished instantiating model with {} architecture".format(
                self.cfg.MODEL.MODEL_ARCH
            )
        )
        self.logger.info("Adding plugins after constructing model")
        self.addPluginsToModel()

    # Add model hooks here, since it is a ModelAbstract
    def addPlugins(self, plugin_class_list: List[Type[ModelPlugin]], **kwargs):
        """add plugins to the plugins queue...
        ideally this is called before add model
        then add model can add the plugins in the queue to the model...
        """
        for plugin in plugin_class_list:
            self.plugins.append(plugin)
        if self.model is not None:
            # model is defined. We will add plugins to queue, and then call the model's add plugin bit...
            self.logger.info("Model already constructed. Adding plugins.")
            self.addPluginsToModel()

    def addPluginsToModel(self):
        for plugin in self.plugins:
            self.model.addPlugin(plugin, 
                            plugin_name = self.cfg.MODEL_PLUGIN[plugin.name].PLUGIN_NAME, 
                            plugin_kwargs = self.cfg.MODEL_PLUGIN[plugin.name].PLUGIN_KWARGS)
            
    def addModelBuilder(
        self, model_builder: Type[Callable], model_config: ModelConfig = None
    ):
        self._modelBuilderQueue = model_builder
        self._modelBuilderQueueFlag = True
        if model_config is not None:
            self._modelConfigQueue = model_config
            self._modelConfigQueueFlag = True

    def addModel(self, model: ModelAbstract):
        self._modelQueue = model
        self._modelQueueFlag = True

    def addModelClass(self, model_class: Type[ModelAbstract], **kwargs):
        self._modelClassQueue = model_class
        self._modelClassQueueFlag = True
        self._modelArgsQueue = kwargs
        if len(self._modelArgsQueue) > 0:
            self._modelArgsQueueFlag = True

    def loadWeights(self):
        """If in `test` mode, load weights from weights path. Otherwise, partially load what is possible from given weights path, if given.
        Note that for training mode, weights are downloaded from pytoch to be loaded if pretrained weights are desired.
        """
        if self.mode == "test":
            if self.weights is None:
                self.logger.info(
                    "No saved model weights provided. Inferring weights path."
                )
                if self.load_epoch is not None and type(self.load_epoch) is int:
                    self.weights = self.getModelWeightsFromEpoch(
                        self.load_epoch
                    )
                    self.logger.info(
                        "Using weights from provided epoch %i, at path %s."
                        % (self.load_epoch, self.weights)
                    )
                else:
                    if self.previous_stop < 0:
                        self.logger.info(
                            "No previous stop exists. Not loading weights."
                        )
                    else:
                        self.weights = self.getModelWeightsFromEpoch(
                            self.previous_stop
                        )
                        self.logger.info(
                            "Using weights from last saved epoch %i, at"
                            " path %s." % (self.previous_stop, self.weights)
                        )
            if self.weights is not None:    # we have this, because previous if-block might update weights path
                self.model.load_state_dict(torch.load(self.weights))
        else:
            if self.weights is None:
                self.logger.info("No saved model weights provided.")
            else:
                if (
                    self.weights != ""
                ):  # Load weights if train and starting from a another model base...
                    self.logger.info(
                        "Commencing partial model load from {}".format(
                            self.weights
                        )
                    )
                    self.model.partial_load(self.weights)
                    self.logger.info(
                        "Completed partial model load from {}".format(
                            self.weights
                        )
                    )
                else:
                    self.logger.info(
                        "In train mode, but no weights provided to load into"
                        " model. This means either model will be initialized"
                        " randomly, or weights loading occurs inside the model."
                    )

    def getModelSummary(self, input_size=None, dtypes=None, **kwargs):
        """Gets the model summary using `torchinfo` and saves it as a ModelStatistics object"""
        # add bookkeeping
        # does config has input size? if it does use that, but prepend batch size .
        # if it doesn't have it, use input size in arguments
        self.model.cuda()
        # change below statement according to line 722
        # default for input size is None/null
        try:
            if input_size is None:
                input_size = (
                    self.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE,
                    self.cfg.TRAIN_TRANSFORMATION.INPUT_SIZE,
                ) # INPUT SIZE SHOULD HAVE A VALUE
            print("INPUT SIZE ==== ",input_size)

            self.model_summary = summary(
                self.model,
                input_size=input_size,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                depth=3,
                mode="train",
                verbose=0,
                dtypes=dtypes,
            )
            self.logger.info(str(self.model_summary))
        except KeyboardInterrupt:   # TODO check which exception is actually raised in summary and maybe have a better way to deal with this...
            raise
        except Exception as e:
            import traceback
            self.logger.info("Model Summary retured the following error:")
            self.logger.info(traceback.format_exc())


    #------------------------------GENERAL ADD--------------------------------------------------------------------
    def add(self, file_or_module_path):
        # We are given a file, which contains several class loaded through decorators when the file is imported.
        imported_file = path_import(file_or_module_path)
        # Once we have imported, the files are registered in ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS
        lookup_path = os.path.abspath(file_or_module_path)
        for keyvalue in ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path]:
            if keyvalue in self.decorator_reference:
                self.decorator_reference[keyvalue](
                    ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue]
                )
            else:
                warnings.warn(
                    "keyvalue %s in REGISTERED_EDNA_COMPONENTS %s is not available in self.decorator_reference. Not adding."%(keyvalue, 
                    str(ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue]))
                )
    # ----------------------------------------------   DATAREADERS   ----------------------------------------------
    def addCrawlerClass(self, crawler_class: Type[Crawler], **kwargs):
        """Adds a crawler class to the EdnaML `apply()` queue. This will be applied to the configuration when calling `apply()`

        The crawler class is added to the internal datareader instance through `apply()`. Then,
        the buildTrainDataloader() and buildTestDataloader() can take instances of this class
        to crawl the dataset and yield batches.

        Args:
            crawler_class (Type[Crawler]): _description_
        """
        self._crawlerClassQueue = crawler_class
        self._crawlerArgsQueue = kwargs
        if len(self._crawlerArgsQueue) > 0:
            self._crawlerArgsQueueFlag = True
        self._crawlerClassQueueFlag = True

    def addCrawler(self, crawler_instance: Crawler):
        """Adds a crawler instance to the EdnaML `apply()` queue.

        Args:
            crawler_instance (Crawler): _description_
        """
        self._crawlerInstanceQueue = crawler_instance
        self._crawlerInstanceQueueFlag = True

    def addGeneratorClass(self, generator_class: Type[Generator], **kwargs):
        """Adds a generator class to the EdnaML `apply()` queue. This will be applied to the configuration when calling `apply()`

        The generator class is added to the internal datareader instance through `apply()`. Then,
        the buildTrainDataloader() and buildTestDataloader() can take instances of this class
        to crawl the dataset and yield batches.

        Args:
            generator_class (Type[Generator]): _description_
        """
        self._generatorClassQueue = generator_class
        self._generatorArgsQueue = kwargs
        if len(self._generatorArgsQueue) > 0:
            self._generatorArgsQueueFlag = True
        self._generatorClassQueueFlag = True

    def addTrainGenerator(self, generator: Generator):
        """Adds a generator instance to the EdnaML `apply()` queue.

        Args:
            generator_class (Type[Generator]): _description_
        """
        self._trainGeneratorInstanceQueue = generator
        self._trainGeneratorInstanceQueueFlag = True

    def addTestGenerator(self, generator: Generator):
        """Adds a generator instance to the EdnaML `apply()` queue.

        Args:
            generator_class (Type[Generator]): _description_
        """
        self._testGeneratorInstanceQueue = generator
        self._testGeneratorInstanceQueueFlag = True

    def buildDataloaders(self):
        """Sets up the datareader classes and builds the train and test dataloaders"""

        data_reader: Type[DataReader] = locate_class(
            package="ednaml",
            subpackage="datareaders",
            classpackage=self.cfg.EXECUTION.DATAREADER.DATAREADER,
        )
        data_reader_instance = data_reader()
        self.logger.info("Reading data with DataReader %s" % data_reader_instance.name)
        self.logger.info("Default CRAWLER is %s"%data_reader_instance.CRAWLER)
        self.logger.info("Default DATASET is %s"%data_reader_instance.DATASET)
        self.logger.info("Default GENERATOR is %s"%data_reader_instance.GENERATOR)
        # Update the generator...if needed
        if self._generatorClassQueueFlag:
            self.logger.info("Updating GENERATOR to queued class %s"%self._generatorClassQueue.__name__)
            data_reader_instance.GENERATOR = self._generatorClassQueue
            if self._generatorArgsQueueFlag:
                self.cfg.EXECUTION.DATAREADER.GENERATOR_ARGS = (
                    self._generatorArgsQueue
                )
        else:
            if (
                self.cfg.EXECUTION.DATAREADER.GENERATOR is not None
            ):
                self.logger.info("Updating GENERATOR using config specification to %s"%self.cfg.EXECUTION.DATAREADER.GENERATOR)
                data_reader_instance.GENERATOR = locate_class(
                    package="ednaml",
                    subpackage="generators",
                    classpackage=self.cfg.EXECUTION.DATAREADER.GENERATOR,
                )

        if self._crawlerClassQueueFlag: #here it checkes whether class flag is set, if it is then replace the build in class with custom class
            self.logger.info("Updating CRAWLER to %s"%self._crawlerClassQueue.__name__)
            data_reader_instance.CRAWLER = self._crawlerClassQueue
            if self._crawlerArgsQueueFlag: #check args also
                self.cfg.EXECUTION.DATAREADER.CRAWLER_ARGS = (
                    self._crawlerArgsQueue
                )

        if self._crawlerInstanceQueueFlag:
            self.crawler = self._crawlerInstanceQueue
        else:
            self.crawler = self._buildCrawlerInstance(
                data_reader=data_reader_instance
            )

        self.buildTrainDataloader(data_reader_instance, self.crawler)
        self.buildTestDataloader(data_reader_instance, self.crawler)

    def _buildCrawlerInstance(self, data_reader: DataReader) -> Crawler:
        """Builds a Crawler instance from the data_reader's provided crawler class in `data_reader.CRAWLER`

        Args:
            data_reader (DataReader): A DataReader class

        Returns:
            Crawler: A Crawler instanece for this experiment
        """
        return data_reader.CRAWLER(
            logger=self.logger, **self.cfg.EXECUTION.DATAREADER.CRAWLER_ARGS
        )

    def buildTrainDataloader(
        self, data_reader: DataReader, crawler_instance: Crawler
    ):
        """Builds a train dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        if self._trainGeneratorInstanceQueueFlag:
            self.train_generator: Generator = self._trainGeneratorInstanceQueue
        else:
            print(self.cfg.TRAIN_TRANSFORMATION)
            if self.mode != "test":
                self.train_generator: Generator = data_reader.GENERATOR( ##imp -- initialize generator
                    logger=self.logger,
                    gpus=self.gpus,
                    transforms=self.cfg.TRAIN_TRANSFORMATION,# train_transforms.args ## not imp.. all arguments are in args -- args is attribute which is storoing a dictionary
                    mode="train",
                    **self.cfg.EXECUTION.DATAREADER.GENERATOR_ARGS
                )

                self.train_generator.build( ## imp -- calls build method inside generator class
                    crawler_instance, 
                    batch_size=self.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE, 
                    workers=self.cfg.TRAIN_TRANSFORMATION.WORKERS,
                    **self.cfg.EXECUTION.DATAREADER.DATASET_ARGS
                )
            else:
                self.train_generator = Generator()
        if self.mode != "test":
            self.logger.info(
                "Generated training data generator with %i trainnig data points"
                % len(self.train_generator.dataset)
            )
            self.labelMetadata = self.train_generator.num_entities
            self.logger.info(
                "Running classification model with classes: %s"
                % str(self.labelMetadata.metadata)
            )
        else:
            self.logger.info(
                "Skipped generating training data, because EdnaML is in test"
                " mode."
            )

    def buildTestDataloader(
        self, data_reader: DataReader, crawler_instance: Crawler
    ):
        """Builds a test dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        if self._testGeneratorInstanceQueueFlag:
            self.test_generator: Generator = self._testGeneratorInstanceQueue
        else:
            self.test_generator: Generator = data_reader.GENERATOR(
                logger=self.logger,
                gpus=self.gpus,
                transforms=self.cfg.TEST_TRANSFORMATION,
                mode="test",
                **self.cfg.EXECUTION.DATAREADER.GENERATOR_ARGS
            )
            self.test_generator.build( 
                crawler_instance,
                batch_size=self.cfg.TEST_TRANSFORMATION.BATCH_SIZE,
                workers=self.cfg.TEST_TRANSFORMATION.WORKERS,
                **self.cfg.EXECUTION.DATAREADER.DATASET_ARGS
            )

        if self.mode == "test":
            self.labelMetadata = self.test_generator.num_entities
        self.logger.info("Generated test data/query generator")

    def buildLogger(
        self,
        logger: logging.Logger = None,
        add_filehandler: bool = True,
        add_streamhandler: bool = True,
        **kwargs
    ) -> logging.Logger:
        """Builds a new logger or adds the correct file and stream handlers to
        existing logger if it does not already have them.

        Args:
            logger (logging.Logger, optional): A logger.. Defaults to None.
            add_filehandler (bool, optional): Whether to add a file handler to the logger. If False, no file is created or appended to. Defaults to True.
            add_streamhandler (bool, optional): Whether to add a stream handler to the logger. If False, logger will not stream to stdout. Defaults to True.

        Returns:
            logging.Logger: A logger with file and stream handlers.
        """
        loggerGiven = True
        if logger is None:
            logger = logging.Logger(self.saveMetadata.MODEL_SAVE_FOLDER)
            loggerGiven = False

        logger_save_path = os.path.join(
            self.saveMetadata.MODEL_SAVE_FOLDER,
            self.saveMetadata.LOGGER_SAVE_NAME,
        )
        # Check for backup logger
        if self.cfg.SAVE.LOG_BACKUP:
            backup_logger = os.path.join(
                self.saveMetadata.CHECKPOINT_DIRECTORY,
                self.saveMetadata.LOGGER_SAVE_NAME,
            )
            if os.path.exists(backup_logger) and add_filehandler:
                print(
                    "Existing log file exists at network backup %s. Will"
                    " attempt to copy to local directory %s."
                    % (backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER)
                )
                shutil.copy2(backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER)
        if os.path.exists(logger_save_path) and add_filehandler:
            print(
                "Log file exists at %s. Will attempt to append there."
                % logger_save_path
            )

        streamhandler = False
        filehandler = False

        if logger.hasHandlers():
            for handler in logger.handlers():
                if isinstance(handler, logging.StreamHandler):
                    streamhandler = True
                if isinstance(handler, logging.FileHandler):
                    if handler.baseFilename == os.path.abspath(
                        logger_save_path
                    ):
                        filehandler = True

        if not loggerGiven:
            logger.setLevel(logging.DEBUG)

        if not filehandler and add_filehandler:
            fh = logging.FileHandler(
                logger_save_path, mode="a", encoding="utf-8"
            )
            fh.setLevel(self.logLevels[self.verbose])
            formatter = logging.Formatter(
                "%(asctime)s %(message)s", datefmt="%H:%M:%S"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if not streamhandler and add_streamhandler:
            cs = logging.StreamHandler()
            cs.setLevel(self.logLevels[self.verbose])
            cs.setFormatter(
                logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(cs)

        return logger

    def log(self, message: str, verbose: int = 3):
        """Logs a message. TODO needs to be fixed.

        Args:
            message (str): Message to log
            verbose (int, optional): Logging verbosity. Defaults to 3.
        """
        self.logger.log(self.logLevels[verbose], message)

    def printConfiguration(self):
        """Prints the EdnaML configuration"""
        self.logger.info("*" * 40)
        self.logger.info("")
        self.logger.info("")
        self.logger.info("Using the following configuration:")
        self.logger.info(self.cfg.export())
        self.logger.info("")
        self.logger.info("")
        self.logger.info("*" * 40)

    def getModelWeightsFromEpoch(self, epoch=0):
        model_load = (
            self.saveMetadata.MODEL_SAVE_NAME + "_epoch%i" % epoch + ".pth"
        )
        if self.cfg.SAVE.DRIVE_BACKUP:
            self.logger.info("Loading model from drive backup.")
            model_load_path = os.path.join(
                self.saveMetadata.CHECKPOINT_DIRECTORY, model_load
            )
        else:
            self.logger.info("Loading model from local backup.")
            model_load_path = os.path.join(
                self.saveMetadata.MODEL_SAVE_FOLDER, model_load
            )
        if os.path.exists(model_load_path):
            return model_load_path
        return None

    def setModelWeightsFromEpoch(self, epoch=0):
        """Sets the internal model weights path using the provided epoch value.
        If there is no corresponding weights path to this epoch value, then
        sets the weights path to `None`

        Args:
            epoch (int, optional): The weights saved at this epoch. Defaults to 0.
        """
        self.weights = self.getModelWeightsFromEpoch(epoch=epoch)

    def loadEpoch(self, epoch=0):
        """Loads weights saved at a specific epoch into the current stored model
        in `self.model`. If no weights are saved for that epoch, logs this and
        does nothing.

        For the provided epoch, `loadEpoch` will check local save directory as
        well as backup save directory for file matching model name constructor
        from `self.saveMetadata`

        Args:
            epoch (int, optional): The epoch to load. If None, then will not do anything.
                If weights corresponding to this epoch do not exist, will not do anything. Defaults to 0.
        """
        model_load_path = self.getModelWeightsFromEpoch(epoch=epoch)
        if model_load_path is not None:
            self.model.load_state_dict(torch.load(model_load_path))
            self.logger.info(
                "Finished loading model state_dict from %s" % model_load_path
            )
        else:
            self.logger.info("No saved weights provided.")

    def deleteLocal(self):
        import shutil
        shutil.rmtree(self.saveMetadata.MODEL_SAVE_FOLDER)

    def setBackupToFalse(self):
        self.cfg.SAVE.DRIVE_BACKUP = False
