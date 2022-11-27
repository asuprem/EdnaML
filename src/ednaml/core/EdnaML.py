import importlib
import os, logging, glob, re
from types import MethodType
from typing import Callable, Dict, List, Type, Union
import warnings
from torchinfo import ModelStatistics
from ednaml import storage
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.config.LossConfig import LossConfig
from ednaml.config.ModelConfig import ModelConfig
from ednaml.core import EdnaMLBase, EdnaMLContextInformation
from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators import Generator
from ednaml.logging import LogManager
from ednaml.loss.builders import LossBuilder
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.optimizer import BaseOptimizer
from ednaml.optimizer.StandardLossOptimizer import StandardLossOptimizer
from ednaml.loss.builders import ClassificationLossBuilder
from ednaml.plugins.ModelPlugin import ModelPlugin
from ednaml.trainer.BaseTrainer import BaseTrainer
from ednaml.storage import BaseStorage, StorageManager
from ednaml.utils import ERSKey, ExperimentKey, StorageArtifactType, StorageKey, locate_class, path_import
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
    model: ModelAbstract
    loss_function_array: List[LossBuilder]
    loss_optimizer_array: List[torch.optim.Optimizer]
    optimizer: List[torch.optim.Optimizer]
    scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    trainer: BaseTrainer
    crawler: Crawler
    train_generator: Generator
    test_generator: Generator
    cfg: EdnaMLConfig
    config: Union[str,List[str]]    # list of paths to configuration files
    decorator_reference: Dict[str,Type[MethodType]]
    plugins: Dict[str, ModelPlugin]
    storage: Dict[str, BaseStorage]
    storage_classes: Dict[str, Type[BaseStorage]]
    storageManager: StorageManager
    logManager: LogManager
    logger: logging.Logger
    experiment_key: ExperimentKey

    context_information: EdnaMLContextInformation

    def __init__(
        self,
        config: Union[List[str], str] = "",
        mode: str = "train",
        weights: str = None,
        **kwargs
    ) -> None:
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
            load_step: If you want EdnaML to load a model from a specific saved step. Epoch must be provided. The path construction will be inferred from the config file
            load_run: If you want EdnaML to use a specific run, instead of the most recent one
            add_filehandler: If you want logger to write to log file
            add_streamhandler: If you want logger to write to stdout
            logger_save_name: If you want logger name to be something other than default constructed name from config
            test_only: Under this condition, if in test mode, only model and dataloaders will be created. Optimizers, schedulers will be empty.

        """
        super().__init__()
        if type(config) is str:
            self.config = config
        elif type(config) is list:
            self.config = []
            for cpath in config:
                if type(cpath) is str:
                    self.config.append(cpath)
                elif type(cpath) is list:
                    self.config += cpath
                else:
                    ValueError("config MUST be list or string")
        else:
            raise ValueError("config MUST be list or string")

        self.mode = mode
        self.weights = weights
        self.gpus = torch.cuda.device_count()
        # TODO Deal with extensions
        if type(self.config) is str:
            self.cfg = EdnaMLConfig([self.config], **kwargs)
        else:
            self.cfg = EdnaMLConfig(self.config, **kwargs)

        self.experiment_key = ExperimentKey(self.cfg.SAVE.MODEL_CORE_NAME, 
                                                self.cfg.SAVE.MODEL_VERSION,
                                                self.cfg.SAVE.MODEL_BACKBONE,
                                                self.cfg.SAVE.MODEL_QUALIFIER)
        # Handled by storage manager
        #os.makedirs(self.saveMetadata.MODEL_SAVE_FOLDER, exist_ok=True)
        # We create a cached directory for temporary logs while EdnaML starts up...
        # Or maybe we create the logger inside storage...!
        log_manager_class = "FileLogManager"
        log_kwargs = {"log_level": 10}  # 10 is DEBUG
        log_manager_class: Type[LogManager] = locate_class(subpackage="logging", classfile=log_manager_class, classpackage=log_manager_class)
        self.logManager = log_manager_class(experiment_key=self.experiment_key, **log_kwargs)
        self.logManager.apply()
        self.logger = self.logManager.getLogger()
        

        self.load_run: int = kwargs.get("load_run", None)
        self.load_epoch: int = kwargs.get("load_epoch", None)
        self.load_step: int = kwargs.get("load_step", None)
        self.test_only: bool = kwargs.get("test_only", False)
        if self.test_only and self.mode == "train":
            raise ValueError(
                "Cannot have `test_only` and be in training mode. Switch to"
                " `test` mode."
            )
        self.resetQueueArray:List[MethodType] = [self.resetCrawlerQueues, self.resetGeneratorQueues, self.resetModelQueues, self.resetOptimizerQueues, self.resetLossBuilderQueue, self.resetStorageQueues, self.resetTrainerQueue]
        self.resetQueueArray += self.addResetQueues()
        self.resetQueues()

        self.decorator_reference = {
            "crawler": self.addCrawlerClass,
            "model": self.addModelClass,
            "trainer": self.addTrainerClass,
            "storage": self.addStorageClass,
            "generator": self.addGeneratorClass,
            "model_plugin": self.addPlugins,
        }
        self.log("Initialized empty Context Object")
        self.context_information = EdnaMLContextInformation()

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
    def resetStorageQueues(self):
        self._storageInstanceQueueFlag = False
        self._storageClassQueueFlag = False
        self._storageInstanceQueue = None
        self._storageClassQueue = None


    def resetQueues(self):
        """Resets the `apply()` queue"""
        self.debug("Resetting declarative queues.")
        for queue_function in self.resetQueueArray:
            queue_function()

    def recordVars(self):
        """recordVars() completes initial setup, allowing you to proceed with the core ml pipeline
        of dataloading, model building, etc
        """

        self.epochs = self.cfg.EXECUTION.EPOCHS
        self.skipeval = self.cfg.EXECUTION.SKIPEVAL
        self.step_verbose = self.cfg.LOGGING.STEP_VERBOSE
        self.test_frequency = self.cfg.EXECUTION.TEST_FREQUENCY
        self.fp16 = self.cfg.EXECUTION.FP16

    def downloadModelWeights(self):
        """Downloads model weights specified in the configuration if `weights` were not passed into EdnaML and if model weights are supported.

        TODO -- do not throw error for no weights or missing base, if this is a new architecture to be trained from scratch
        Raises:
            Warning: If there are no pre-downloaded weights, and the model architecture is unsupported
        """
        if self.weights is not None:
            self.log(
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
                self.log(
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
        # Print the current configuration state
        self.printConfiguration()
        # Build the Storage Manager
        self.log("[APPLY] Building StorageManager")
        self.buildStorageManager()
        # Build the storage backends that StorageManager can use
        self.log("[APPLY] Adding Storages")
        self.buildStorage()
        # Set the RunKey for this ExperimentKey
        # Upload the configuration for this Run
        self.log("[APPLY] Setting tracking run")
        self.setTrackingRun(**kwargs)
        # Obtain the latest weights path. This will be the continuation epoch, regardless of whether weights are provided or not, during training.
        self.log("[APPLY] Setting latest StorageKey")
        self.setLatestStorageKey()
        # Set up the LogManager. LogManager either writes to file OR logs to some log server by itself.
        self.log("[APPLY] Updating Logger with Latest ERSKey")
        self.updateLoggerWithERS()
        # Download pre-trained weights, if such a link is provided in built-in model paths
        self.log("[APPLY] Downloading pre-trained weights, if available")
        self.downloadModelWeights()
        # Build the data loaders
        self.log("[APPLY] Building dataloaders")
        self.buildDataloaders()
        # Build the model
        self.log("[APPLY] Building model")
        self.buildModel()
        # For test mode, load the most recent weights using LatestStorageKey unless explicit epoch-step provided
        # For train mode, load weights iff provided. Otherwise, Trainer will take care of it.
        # For EdnaDeploy, load the most recent weights using LatestStorageKey unless explicit epoch-step provided.
        self.log("[APPLY] Loading latest weights, if available")
        self.loadWeights()
        # Generate a model summary
        self.log("[APPLY] Generating summary")
        self.getModelSummary(**kwargs)
        # Build the optimizer
        self.log("[APPLY] Building optimizer, scheduler, and losses")
        self.buildOptimizer()
        # Build the scheduler
        self.buildScheduler()
        # Build the loss array
        self.buildLossArray()
        # Build the Loss optimizers, for learnable losses
        self.buildLossOptimizer()
        # Build the loss schedulers, for learnable losses
        self.buildLossScheduler()
        # Build the trainer
        self.log("[APPLY] Building trainer")
        self.buildTrainer()
        # Reset the queues. Maybe clear the plugins and storage queues as well??
        self.resetQueues()

    def buildStorageManager(self):  # TODO after I get a handle on the rest...
        self.storageManager = StorageManager(
            logger = self.logger,
            cfg = self.cfg,
            experiment_key = self.experiment_key,
            storage_manager_mode="loose"
        )

    def addStorage(self, storage_class_dict: Dict[str, BaseStorage]):
        """Adds a list of already instantiated storages classes, with their storage names
        to the internal storage dictionary. The dictionary should be of the form:

        {
            storage_name: InstantiatedStorageClassObject, ...
        }

        `storage_name` MUST match referenced storage in the configuration.

        """
        
        for storage_name in storage_class_dict:
            self.debug("Added custom storage %s with `storage_name` %s"%(storage_class_dict[storage_name].__class__.__name__, storage_name))
            self.storage[storage_name] = storage_class_dict[storage_name]
        # We don't need queues here because Storages are instantiated lazily, 
        # i.e. only if they are needed from the configuration
    
    def addStorageClass(self, storage_class_list: List[Type[BaseStorage]]):
        """Adds a list of storage classes to the queue.
        Storage classes are in a list due to how decorators themselves work.

        We can extract the storage name directly from the class.

        Args:
            storage_class_list (List[Type[BaseStorage]]): List of storage classes inherited from BaseStorage
        """
        
        for storage in storage_class_list:
            self.debug("Added custom storage: %s"%storage.__name__)
            self.storage_classes[storage.__name__] = storage
        # We don't need queues here because Storages are instantiated lazily, 
        # i.e. only if they are needed from the configuration

    def buildStorage(self):  
        # We look at config to see what storages we need to load
        # Then we check their classes: if they are first, we check in decorated loaded list.
        # Then we check built-in

        # Load them into a dictionary

        # So, first we check if a class or instance was directly passed.
        # If nothing has been passed, then we try to locate a built-in version based on storage.type
        # If that doesn't work, then we can throw an error...
        for storage_element in self.cfg.STORAGE:
            if self.cfg.STORAGE[storage_element].STORAGE_NAME in self.storage:
                self.debug("Skipping storage with name %s, already exists"%storage_element)
            else:
                storage_class_name = self.cfg.STORAGE[storage_element].STORAGE_CLASS
                if storage_class_name in self.storage_classes:
                    storage_class_reference: Type[BaseStorage] = self.storage_classes[storage_class_name]
                    self.log(
                    "Loaded {} from {} to build Storage".format(
                        self.cfg.STORAGE[storage_element].STORAGE_CLASS, "storage-list"
                    )
                    )
                else:
                    storage_class_reference: Type[BaseStorage] = locate_class(subpackage="storage", classpackage=storage_class_name)
                    self.log(
                    "Loaded {} from {} to build Storage".format(
                        self.cfg.STORAGE[storage_element].STORAGE_CLASS, "ednaml.storage"
                    )
                    )
                self.storage[self.cfg.STORAGE[storage_element].STORAGE_NAME] = storage_class_reference(storage_name=self.cfg.STORAGE[storage_element].STORAGE_NAME,
                                                                                                            storage_url=self.cfg.STORAGE[storage_element].STORAGE_URL,
                                                                                                            experiment_key=self.experiment_key,
                                                                                                            **self.cfg.STORAGE[storage_element].STORAGE_ARGS)


    def train(self, **kwargs):
        self.trainer.train(ers_key = self.storageManager.getLatestERSKey(), **kwargs)  #

    def eval(self):
        return self.trainer.evaluate()

    def addTrainerClass(self, trainerClass: Type[BaseTrainer]):
        self.debug("Added trainer class: %s"%trainerClass.__name__)
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
            self.log(
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
                storage=self.storage,
                context = self.context_information,
                **self.cfg.EXECUTION.TRAINER_ARGS
            )
            # TODO -- change save_backup, backup_directory stuff. These are all in Storage.... We just need the model save name...
            self.trainer.apply(
                step_verbose=self.cfg.LOGGING.STEP_VERBOSE,
                #save_frequency=self.cfg.SAVE.SAVE_FREQUENCY,
                #step_save_frequency = self.cfg.SAVE.STEP_SAVE_FREQUENCY,
                test_frequency=self.cfg.EXECUTION.TEST_FREQUENCY,
                #save_directory=self.saveMetadata.MODEL_SAVE_FOLDER,
                #save_backup=self.cfg.SAVE.DRIVE_BACKUP,
                #backup_directory=self.saveMetadata.CHECKPOINT_DIRECTORY,
                storage_manager = self.storageManager,
                gpus=self.gpus,
                fp16=self.cfg.EXECUTION.FP16
            )

    def setTrackingRun(self, **kwargs):
        self.storageManager.setTrackingRun(storage_dict=self.storage, 
                                            tracking_run = kwargs.get("tracking_run", None), 
                                            new_run = kwargs.get("new_run", False))

    def uploadConfig(self, **kwargs):
        self.storageManager.upload(storage_dict=self.storage, ers_key=self.storageManager.getNextERSKey(artifact=StorageArtifactType.CONFIG))

    def updateLoggerWithERS(self):
        """Download the existing log file from remote, if possible, so that our LogManager can append to it.
        """
        self.log("Retrieving latest ERSKey")
        ers_key = self.storageManager.getLatestERSKey(artifact=StorageArtifactType.LOG)
        if ers_key.storage.epoch == -1:
            self.log("Latest ERSKey's StorageKey is empty. Resetting StorageKey component.")
            ers_key = self.storageManager.getERSKey(epoch=0,step=0,artifact_type=StorageArtifactType.LOG)
        # If this is a new experiment, the latest_ers_key, before anything has started, is set at -1/-1
        self.log(ers_key)

        success = self.storageManager.download(
            storage_dict=self.storage,
            ers_key=ers_key
        )
        if success:
            self.log("Downloaded remote log to %s"%self.storageManager.getLocalSavePath(ers_key=ers_key))
        else:
            self.log("No remote logger exists. Will create new log file at above ERSKey")
        self.logManager.updateERSKey(ers_key=ers_key,
                        file_name=self.storageManager.getLocalSavePath(ers_key=ers_key))



    def setLatestStorageKey(self):
        """Query the storages to obtain the last epoch-step pair when something was saved. Save this as the latest_storage_key
        StorageManager.

        We query MODEL only
        """
        self.storageManager.setLatestStorageKey(
            storage_dict=self.storage,
            artifact=StorageArtifactType.MODEL,
        )


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
            self.log(
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
                self.log(
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
            self.log("Built optimizer")

    def buildScheduler(self):
        """Builds the scheduler for the model"""
        if self.test_only:
            self.scheduler = []
            self.log(
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
            self.log("Built scheduler")



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
            self.log(
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
            self.log("Built loss function")

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
            self.log(
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
            self.log("Built loss optimizer")

    def buildLossScheduler(self):
        """Builds the scheduler for the loss functions, if the functions have learnable parameters and corresponding optimizer."""
        if self.test_only:
            self.loss_scheduler = []
            self.log(
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
                self.log("Built loss scheduler")

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
        self.log(
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
        self.log(
            "Finished instantiating model with {} architecture".format(
                self.cfg.MODEL.MODEL_ARCH
            )
        )
        self.model._logger = self.logger
        self.log("Adding plugins after constructing model")
        self.addPluginsToModel()

    # Add model hooks here, since it is a ModelAbstract
    def addPlugins(self, plugin_class_list: List[Type[ModelPlugin]], **kwargs):
        """add plugins to the plugins class queue...
        ideally this is called before add model
        then add model can add the plugins in the queue to the model...

        NOTE -- any plugins are added through the config file, specifically, with the plugin arguments. 
        This is because multiple plugins can use the same class, but variations of it.
        For example, one might want to train KMP under l2 and cos, and use then simultaneously...

        So, plugins class queue exists only as a lookup for plugin classes...if a correspondiing enttry is NOT in MODEL_PLUGINS in the config,
        it will never be added into Edna/Deploy...
        """
        for plugin in plugin_class_list:
            self.debug("Added custom plugin: %s"%plugin.__name__)
            self.plugins[plugin.__name__] = plugin  # order matters; can replace...though shouldn't matter too much...
        
        # The follow will not happen, logically, i think, because model is ONLY defined in self.buildModel(), and by that point, one is already calling apply()
        # Adding plugins after the fact...????? Don't think so...
        # And in any case, we should make apply() truely declarative, so that one can simply apply() again after adding plugins
        #if self.model is not None:
        #    # model is defined. We will add plugins to queue, and then call the model's add plugin bit...
        #    self.log("Model already constructed. Adding plugins.")
        #    self.addPluginsToModel()

    def addPluginsToModel(self):
        # Here, we iterate through plugins IN the config file, then import their classes, and add them to the model...
        for plugin_save_name in self.cfg.MODEL_PLUGIN:
            # Now, we need to locate the plugin class...check first if it is in plugins...
            if self.cfg.MODEL_PLUGIN[plugin_save_name].PLUGIN not in self.plugins:
                # now we need to locate the class in builtins
                plugin = locate_class(
                    subpackage="plugins",
                    classpackage=self.cfg.MODEL_PLUGIN[plugin_save_name].PLUGIN,
                    classfile=self.cfg.MODEL_PLUGIN[plugin_save_name].PLUGIN
                )
            else:
                plugin = self.plugins[self.cfg.MODEL_PLUGIN[plugin_save_name].PLUGIN]
            
            self.model.addPlugin(
                plugin = plugin,
                plugin_name = plugin_save_name,
                plugin_kwargs = self.cfg.MODEL_PLUGIN[plugin_save_name].PLUGIN_KWARGS
            )

        #for plugin in self.plugins:
        #    self.model.addPlugin(plugin, 
        #                    plugin_name = self.cfg.MODEL_PLUGIN[plugin.name].PLUGIN_NAME, 
        #                    plugin_kwargs = self.cfg.MODEL_PLUGIN[plugin.name].PLUGIN_KWARGS)
            
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
        self.debug("Added model class: %s"%model_class.__name__)
        self._modelClassQueue = model_class
        self._modelClassQueueFlag = True
        self._modelArgsQueue = kwargs
        if len(self._modelArgsQueue) > 0:
            self._modelArgsQueueFlag = True

    def loadWeights(self):
        """
        # For test mode, load the most recent weights using LatestStorageKey unless explicit epoch-step provided
        # For train mode, load weights iff provided. Otherwise, Trainer will take care of it.
        # For EdnaDeploy, load the most recent weights using LatestStorageKey unless explicit epoch-step provided.
        # If pre-trained weights were downloaded, and LatestStorageKey is -1/-1, then load pre-trained weights.
        """
        if self.mode == "test":
            if self.weights is None:
                self.log(
                    "No saved model weights provided. Inferring weights path from the latest ERSKey:"
                )
                model_ers_key = self.storageManager.getLatestERSKey(artifact=StorageArtifactType.MODEL)
                self.log(model_ers_key)
                # success to check if a model was downloaded or not...
                success = self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].downloadModel(
                    ers_key=model_ers_key,
                    destination_file_name=self.storageManager.getLocalSavePath(model_ers_key)
                )

                if success:
                    self.log("Found model in Storage at provided ERSKey")
                    artifact_ers_key = self.storageManager.getLatestERSKey(artifact=StorageArtifactType.ARTIFACT)
                    self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].downloadModelArtifact(
                        ers_key=artifact_ers_key,
                        destination_file_name=self.storageManager.getLocalSavePath(artifact_ers_key)
                    )
                    self.weights = self.storageManager.getLocalSavePath(model_ers_key)    
                    self.model.load_state_dict(torch.load(self.weights))
                    self.context_information.MODEL_HAS_LOADED_WEIGHTS
                    self.context_information.LOADED_EPOCH = model_ers_key.storage.epoch
                    self.context_information.LOADED_STEP = model_ers_key.storage.step

                    self.log(
                    "Downloaded weights from epoch %i, step %i to local path %s"%(model_ers_key.storage.epoch, model_ers_key.storage.step, self.weights)
                )
                else:
                    self.log(
                    "Could not download weights."
                )
                
        else:
            if self.weights is None:
                self.log("No saved model weights provided. BaseTrainer will load weights")
            else:
                if (
                    self.weights != ""
                ):  # Load weights if train and starting from a another model base...
                    self.log(
                        "Commencing partial model load from {}".format(
                            self.weights
                        )
                    )
                    self.model.partial_load(self.weights)
                    self.context_information.MODEL_HAS_LOADED_WEIGHTS = True
                    self.log(
                        "Completed partial model load from {}".format(
                            self.weights
                        )
                    )
                else:
                    self.log(
                        "In train mode, but no weights provided to load into"
                        " model. This means either model will be initialized"
                        " randomly, or weights loading occurs inside the model/BaseTrainer."
                    )

    def getModelSummary(self, input_size=None, dtypes=None, **kwargs):
        """Gets the model summary using `torchinfo` and saves it as a ModelStatistics object"""
        # add bookkeeping
        # does config has input size? if it does use that, but prepend batch size .
        # if it doesn't have it, use input size in arguments
        # TODO possibly move this into trainer...?
        # or at least deal with potential mlti-gpu scenario...
        if kwargs.get("skip_model_summary", False):
            return
        if self.cfg.LOGGING.INPUT_SIZE is not None:
            if input_size is None:
                input_size = self.cfg.LOGGING.INPUT_SIZE
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
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
            self.log(str(self.model_summary))
        except KeyboardInterrupt:   # TODO check which exception is actually raised in summary and maybe have a better way to deal with this...
            raise
        except Exception as e:
            import traceback
            self.log("Model Summary retured the following error:")
            self.log(traceback.format_exc())


    #------------------------------GENERAL ADD--------------------------------------------------------------------
    def _add(self, file_or_module_path):
        # We are given a file, which contains several class loaded through decorators when the file is imported.
        imported_file = path_import(file_or_module_path)
        # Once we have imported, the files are registered in ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS
        lookup_path = os.path.abspath(file_or_module_path)
        for keyvalue in ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path]:
            if keyvalue in self.decorator_reference:
                if type(ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue]) is list:
                  for i in range(len(ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue])):
                    self.log("Adding a {ftype}, from {src}, with inferred name {inf}".format(
                    ftype=keyvalue,
                    src = lookup_path,
                    inf = ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue][i].__name__
                ))
                else:
                  self.log("Adding a {ftype}, from {src}, with inferred name {inf}".format(
                      ftype=keyvalue,
                      src = lookup_path,
                      inf = ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue].__name__
                  ))
                self.decorator_reference[keyvalue](
                    ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue]
                )
            else:
                warnings.warn(
                    "keyvalue %s in REGISTERED_EDNA_COMPONENTS %s is not available in self.decorator_reference. Not adding."%(keyvalue, 
                    str(ednaml.core.decorators.REGISTERED_EDNA_COMPONENTS[lookup_path][keyvalue]))
                )
    def add(self, file_or_module_path):
        if type(file_or_module_path) is list:
            for file_or_module in file_or_module_path:
                self._add(file_or_module)
        else:
            self._add(file_or_module_path)
        
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
            classpackage=self.cfg.DATAREADER.DATAREADER,
        )
        data_reader_instance = data_reader()
        self.log("Reading data with DataReader %s" % data_reader_instance.name)
        self.log("Default CRAWLER is %s"%data_reader_instance.CRAWLER)
        self.log("Default DATASET is %s"%data_reader_instance.DATASET)
        self.log("Default GENERATOR is %s"%data_reader_instance.GENERATOR)
        # Update the generator...if needed
        if self._generatorClassQueueFlag:
            self.log("Updating GENERATOR to queued class %s"%self._generatorClassQueue.__name__)
            data_reader_instance.GENERATOR = self._generatorClassQueue
            if self._generatorArgsQueueFlag:
                self.cfg.DATAREADER.GENERATOR_ARGS = (
                    self._generatorArgsQueue
                )
        else:
            if (
                self.cfg.DATAREADER.GENERATOR is not None
            ):
                self.log("Updating GENERATOR using config specification to %s"%self.cfg.DATAREADER.GENERATOR)
                data_reader_instance.GENERATOR = locate_class(
                    package="ednaml",
                    subpackage="generators",
                    classpackage=self.cfg.DATAREADER.GENERATOR,
                )

        if self._crawlerClassQueueFlag: #here it checkes whether class flag is set, if it is then replace the build in class with custom class
            self.log("Updating CRAWLER to %s"%self._crawlerClassQueue.__name__)
            data_reader_instance.CRAWLER = self._crawlerClassQueue
            if self._crawlerArgsQueueFlag: #check args also
                self.cfg.DATAREADER.CRAWLER_ARGS = (
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
            logger=self.logger, **self.cfg.DATAREADER.CRAWLER_ARGS
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
            #print(self.cfg.TRAIN_TRANSFORMATION)
            if self.mode != "test":
                self.train_generator: Generator = data_reader.GENERATOR( ##imp -- initialize generator
                    logger=self.logger,
                    gpus=self.gpus,
                    transforms=self.cfg.TRAIN_TRANSFORMATION,# train_transforms.args ## not imp.. all arguments are in args -- args is attribute which is storoing a dictionary
                    mode="train",
                    **self.cfg.DATAREADER.GENERATOR_ARGS
                )

                self.train_generator.build( ## imp -- calls build method inside generator class
                    crawler_instance, 
                    batch_size=self.cfg.TRAIN_TRANSFORMATION.BATCH_SIZE, 
                    workers=self.cfg.TRAIN_TRANSFORMATION.WORKERS,
                    **self.cfg.DATAREADER.DATASET_ARGS
                )
            else:
                self.log(
                "Not creating `train_generator` in `test_only` mode."
                )
                self.train_generator = Generator(logger=self.logger)
        if self.mode != "test":
            self.log(
                "Generated training data generator with %i training data points"
                % len(self.train_generator.dataset)
            )
            self.labelMetadata = self.train_generator.num_entities
            self.log(
                "Running classification model with classes: %s"
                % str(self.labelMetadata.metadata)
            )
        else:
            self.log(
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
                **self.cfg.DATAREADER.GENERATOR_ARGS
            )
            self.test_generator.build( 
                crawler_instance,
                batch_size=self.cfg.TEST_TRANSFORMATION.BATCH_SIZE,
                workers=self.cfg.TEST_TRANSFORMATION.WORKERS,
                **self.cfg.DATAREADER.DATASET_ARGS
            )

        if self.mode == "test":
            self.labelMetadata = self.test_generator.num_entities
        self.log("Generated test data/query generator")

    def printConfiguration(self):
        """Prints the EdnaML configuration"""
        self.log("*" * 40)
        self.log("")
        self.log("Using the following configuration:\n" + self.cfg.export())
        self.log("")
        self.log("*" * 40)

    def getModelWeightsFromStorageKey(self, epoch: int = None, step: int = None, storage_key: StorageKey = None):
        if storage_key is not None:
            epoch = storage_key.epoch
            step = storage_key.step

        if step is None:
            step = -1
        if epoch is None:
            epoch = -1
        ers_key: ERSKey = self.storageManager.getERSKey(epoch = epoch, step = step)

        final_ers_key: ERSKey = None
        # First, if epoch is 0, we get the highest epoch value
        if epoch == -1:
            self.log("No epoch value provided. Searching for latest saved epoch.")
            final_ers_key = self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].getLatestModelEpoch(ers_key)
        else:
            self.log("Using provided `load_epoch` of %s"%ers_key.storage.epoch)
            final_ers_key = ers_key # i.e. preserve the epoch value from the ers_key, since it was not -1
        if final_ers_key is None:   # No models with any epoch exist...
            self.log("No models exist. Not loading any models.")
            return None, None, None
        
        # Next, if there *was* a model, but step is -1, we get the maximum step valued model
        if step == -1:
            self.log("No step value provided. Searching for latest saved step with epoch %s."%final_ers_key.storage.epoch)
            final_ers_key = self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].getLatestModelWithEpoch(final_ers_key)
        else:
            self.log("Searching for model with provided `load_epoch` %s and `load_step` %s"%(final_ers_key.storage.epoch, final_ers_key.storage.step))
            final_ers_key = self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].getKey(final_ers_key)

        # i.e. no weights provided.
        if final_ers_key is None:
            self.log("No models exist at provided `load_epoch` and `load_step`. Not loading any models.")
            return None, None, None
        else:
            model_path = self.storageManager.getLocalSavePath(final_ers_key)
            self.storage[self.storageManager.getStorageNameForArtifact(StorageArtifactType.MODEL)].downloadModel(final_ers_key, model_path)
        self.log("Found `model_path` at provided `load_epoch` and `load_step`: %s"%os.path.basename(model_path))
        return model_path, final_ers_key.storage.epoch, final_ers_key.storage.step
    
    # TODO fix this i.e. harmonize
    def getModelWeightsFromEpoch(self, epoch=0):
        model_load = (
            self.saveMetadata.MODEL_SAVE_NAME + "_epoch%i" % epoch + ".pth"
        )
        if self.cfg.SAVE.DRIVE_BACKUP:
            self.log("Loading model from drive backup.")
            model_load_path = os.path.join(
                self.saveMetadata.CHECKPOINT_DIRECTORY, model_load
            )
        else:
            self.log("Loading model from local backup.")
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
            self.log(
                "Finished loading model state_dict from %s" % model_load_path
            )
        else:
            self.log("No saved weights provided.")

    def deleteLocal(self):
        import shutil
        shutil.rmtree(self.saveMetadata.MODEL_SAVE_FOLDER)

    def setBackupToFalse(self):
        self.cfg.SAVE.DRIVE_BACKUP = False