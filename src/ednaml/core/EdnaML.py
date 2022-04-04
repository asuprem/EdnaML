import importlib
import os, shutil, logging, glob, re, json
from pydoc import locate
from typing import Dict, List, Type
import warnings
from torchinfo import ModelStatistics
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators import ImageGenerator
from ednaml.loss.builders import LossBuilder
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.optimizer import BaseOptimizer
from ednaml.optimizer.StandardLossOptimizer import StandardLossOptimizer
from ednaml.loss.builders import ClassificationLossBuilder
from ednaml.trainer.BaseTrainer import BaseTrainer
from ednaml.utils import locate_class
import ednaml.utils
import torch
from torchinfo import summary
from ednaml.utils.LabelMetadata import LabelMetadata
import ednaml.utils.web
from ednaml.utils.SaveMetadata import SaveMetadata
import logging

"""TODO

Verbosity = 0 -> create no logger and use a dummy that print nothing anywhere
"""


class EdnaML:
    logLevels = {0: logging.NOTSET, 1: logging.ERROR, 2: logging.INFO, 3: logging.DEBUG}
    labelMetadata: LabelMetadata
    modelStatistics: ModelStatistics
    model: ModelAbstract
    loss_function_array: List[LossBuilder]
    loss_optimizer_array: List[torch.optim.Optimizer]
    optimizer: List[torch.optim.Optimizer]
    scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    previous_stop: int
    trainer: BaseTrainer
    crawler: Crawler
    train_generator: ImageGenerator
    test_generator: ImageGenerator
    cfg: EdnaMLConfig

    def __init__(
        self,
        config: str = "config.yaml",
        mode: str = "train",
        weights: str = None,
        logger: logging.Logger = None,
        verbose: int = 2,
    ):
        """Initializes the EdnaML object with the associated config, mode, weights, and verbosity. 
        Sets up the logger, as well as local logger save directory. 

        Args:
            config (str, optional): Path to the Edna config file. Defaults to "config.yaml".
            mode (str, optional): Either `train` or `test`. Defaults to "train".
            weights (str, optional): The path to the weights file. Defaults to None.
            logger (logging.Logger, optional): A logger. If no logger is provided, EdnaML will construct its own. Defaults to None.
            verbose (int, optional): Logging verbosity. Defaults to 2.
        """

        self.config = config
        self.mode = mode
        self.weights = weights
        self.pretrained_weights = None
        self.verbose = verbose

        self.cfg = self.buildConfig(config)
        self.saveMetadata = self.buildSaveMetadata()

        self.gpus = torch.cuda.device_count()
        self.drive_backup = self.cfg.SAVE.DRIVE_BACKUP
        self.previous_stop = 0
        self.epochs = self.cfg.EXECUTION.EPOCHS
        self.skipeval = self.cfg.EXECUTION.SKIPEVAL

        self.step_verbose = self.cfg.LOGGING.STEP_VERBOSE
        self.save_frequency = self.cfg.SAVE.SAVE_FREQUENCY
        self.test_frequency = self.cfg.EXECUTION.TEST_FREQUENCY
        self.fp16 = self.cfg.EXECUTION.FP16

        self.makeSaveDirectories()
        self.logger = self.buildLogger(logger=logger)

        

    def buildConfig(self, config: str, handler="yaml") -> EdnaMLConfig:
        """Builds the internal kaptan configuration object

        Args:
            config (str): Path to config file
            handler (str, optional): Handler for kaptan. Defaults to "yaml".
        """
        return EdnaMLConfig(config)
        # kaptan.Kaptan(handler=handler)
        # self.cfg = self.cfg.import_config(config)

    def setup(self):
        """setup() completes initial setup, allowing you to proceed with the core ml pipeline
        of dataloading, model building, etc
        """
        self.printConfiguration()
        self.downloadModelWeights()

    def downloadModelWeights(self):
        """Downloads model weights specified in the configuration if `weights` were not passed into EdnaML and if model weights are supported.

        TODO -- do not throw error for no weights or missing base, if this is a new architecture to be trained from scratch
        Raises:
            Warning: If there are no pre-downloaded weights, and the model architecture is unsupported
        """
        if self.weights is not None:
            self.logger.info("Not downloading weights. Weights path already provided.")

        if self.mode == "train":
            self._download_weights_from_base(self.cfg.MODEL.MODEL_BASE)
        else:
            if self.weights is None:
                warnings.warn(
                    "Mode is `test` but weights is `None`. This will cause issues when EdnaML attempts to load weights"
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
                "Model %s is not available. Please choose one of the following: %s if you want to load pretrained weights"
                % (model_base, str(model_weights.keys()))
            )

    def quickSetup(self):
        """Performs a `quickstart` set-up of EdnaML
        """
        self.setup()
        self.setPreviousStop()  # TODO -- load weights for previous stop outside of trainer...

        self.buildDataloaders()

        self.buildModel()
        self.loadWeights()
        self.getModelSummary()
        self.buildOptimizer()
        self.buildScheduler()

        self.buildLossArray()
        self.buildLossOptimizer()
        self.buildLossScheduler()

        self.buildTrainer()

    def train(self):
        self.trainer.train(continue_epoch=self.previous_stop)

    def eval(self):
        return self.trainer.evaluate()

    def buildTrainer(self):
        """Builds the EdnaML trainer and sets it up
        """
        ExecutionTrainer: Type[BaseTrainer] = locate_class(subpackage="trainer",classpackage=self.cfg.EXECUTION.TRAINER)
        self.logger.info(
            "Loaded {} from {} to build Trainer".format(
                self.cfg.EXECUTION.TRAINER, "ednaml.trainer"
            )
        )

        self.trainer = ExecutionTrainer(
            model=self.model,
            loss_fn=self.loss_function_array,
            optimizer=self.optimizer,
            loss_optimizer=self.loss_optimizer_array,
            scheduler=self.scheduler,
            loss_scheduler=self.loss_scheduler,
            train_loader=self.train_generator.dataloader,
            test_loader=self.test_generator.dataloader,
            epochs=self.epochs,
            skipeval=self.skipeval,
            logger=self.logger,
            crawler=self.crawler,
            config=self.cfg,
            labels=self.labelMetadata,
        )
        self.trainer.setup(
            step_verbose=self.step_verbose,
            save_frequency=self.save_frequency,
            test_frequency=self.test_frequency,
            save_directory=self.saveMetadata.MODEL_SAVE_FOLDER,
            save_backup=self.drive_backup,
            backup_directory=self.saveMetadata.CHECKPOINT_DIRECTORY,
            gpus=self.gpus,
            fp16=self.fp16,
            model_save_name=self.saveMetadata.MODEL_SAVE_NAME,
            logger_file=self.saveMetadata.LOGGER_SAVE_NAME,
        )

    def setPreviousStop(self):
        """Sets the previous stop
        """
        self.previous_stop = self.getPreviousStop()
    def getPreviousStop(self) -> int:
        """Gets the previous stop, if any, of the trainable model by checking local save directory, as well as a network directory. 
        """
        if self.drive_backup:
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
            self.logger.info("No previous stop detected. Will start from epoch 0")
            return 0
        else:
            self.logger.info(
                "Previous stop detected. Will attempt to resume from epoch %i"
                % self.previous_stop
            )
            return max(previous_stop) + 1

    def buildOptimizer(self):
        """Builds the optimizer for the model
        """
        optimizer_builder: Type[BaseOptimizer] = locate_class(subpackage="optimizer", classpackage=self.cfg.EXECUTION.OPTIMIZER_BUILDER)
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
            OPT.build(self.model.getParameterGroup(self.cfg.OPTIMIZER[idx].OPTIMIZER_NAME)) for idx, OPT in enumerate(OPT_array)
        ]  # TODO deal with singleton vs multiple optimizers...
        self.logger.info("Built optimizer")

    def buildScheduler(self):
        """Builds the scheduler for the model
        """
        self.scheduler = [None] * len(self.cfg.SCHEDULER)
        for idx, scheduler_item in enumerate(self.cfg.SCHEDULER):
            try:  # We first check if scheduler is part of torch's provided schedulers.
                scheduler = locate_class(package="torch.optim", subpackage="lr_scheduler", classpackage=scheduler_item.LR_SCHEDULER)
            except (
                ModuleNotFoundError,
                AttributeError,
            ):  # If it fails, then we try to import from schedulers implemented in scheduler/ folder
                scheduler = locate_class(subpackage="scheduler", classpackage=scheduler_item.LR_SCHEDULER)
            self.scheduler[idx] = scheduler(
                self.optimizer[idx], last_epoch=-1, **scheduler_item.LR_KWARGS
            )
        self.logger.info("Built scheduler")

    def buildLossArray(self):
        """Builds the loss function array using the LOSS list in the provided configuration
        """
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
        loss_optimizer_name_dict = {
            loss_optim_item.OPTIMIZER_NAME: loss_optim_item
            for loss_optim_item in self.cfg.LOSS_OPTIMIZER
        }
        initial_key = list(loss_optimizer_name_dict.keys())[0]
        LOSS_OPT: List[StandardLossOptimizer] = [None] * len(self.loss_function_array)
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
                weight_bias=loss_optimizer_name_dict[lookup_key].WEIGHT_BIAS_FACTOR,
                weight_decay=loss_optimizer_name_dict[lookup_key].WEIGHT_DECAY,
                opt_kwargs=loss_optimizer_name_dict[lookup_key].OPTIMIZER_KWARGS,
            )

        # Note: build returns None if there are no differentiable parameters
        self.loss_optimizer_array = [
            loss_opt.build(loss_builder=self.loss_function_array[idx])
            for idx, loss_opt in enumerate(LOSS_OPT)
        ]
        self.logger.info("Built loss optimizer")

    def buildLossScheduler(self):
        """Builds the scheduler for the loss functions, if the functions have learnable parameters and corresponding optimizer.
        """
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
                    lookup_key = self.loss_function_array[idx].loss_labelname
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
        """Sets model to test mode if EdnaML is in testing mode
        """
        if self.mode == "test":
            self.model.eval()

    def _setModelTrainMode(self):
        """Sets the model to train mode if EdnaML is in training mode
        """
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
        """Builds an EdnaML model using the configuration. If there are pretrained weights, they are provided through the config to initialize the model.
        """
        model_builder = locate_class(subpackage="models", classpackage=self.cfg.MODEL.BUILDER)
        self.logger.info(
            "Loaded {} from {} to build model".format(self.cfg.MODEL.BUILDER, "ednaml.models")
        )

        # model_kwargs = self._covert_model_kwargs()

        # TODO!!!!!!!
        self.model: ModelAbstract = model_builder(
            arch=self.cfg.MODEL.MODEL_ARCH,
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

    def loadWeights(self):
        """If in `test` mode, load weights from weights path. Otherwise, partially load what is possible from given weights path, if given.
        Note that for training mode, weights are downloaded from pytoch to be loaded if pretrained weights are desired.
        """
        if self.mode == "test":
            self.model.load_state_dict(torch.load(self.weights))
        else:
            if self.weights is None:
                self.logger.info("No saved model weights provided.")
            else:
                if (
                    self.weights != ""
                ):  # Load weights if train and starting from a another model base...
                    self.logger.info(
                        "Commencing partial model load from {}".format(self.weights)
                    )
                    self.model.partial_load(self.weights)
                    self.logger.info(
                        "Completed partial model load from {}".format(self.weights)
                    )

    def getModelSummary(self):
        """Gets the model summary using `torchinfo` and saves it as a ModelStatistics object
        """
        self.model.cuda()
        self.model_summary = summary(
            self.model,
            input_size=(
                self.cfg.TRANSFORMATION.BATCH_SIZE,
                self.cfg.TRANSFORMATION.CHANNELS,
                *self.cfg.TRANSFORMATION.SHAPE,
            ),
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
        )
        self.logger.info(str(self.model_summary))

    def buildDataloaders(self):
        """Sets up the datareader classes and builds the train and test dataloaders
        """
        data_reader: Type[DataReader] = locate_class(package="ednaml", subpackage="datareaders", classpackage=self.cfg.EXECUTION.DATAREADER.DATAREADER)
        data_reader_instance = data_reader()
        # data_crawler is now data_reader.CRAWLER
        self.logger.info("Reading data with DataReader %s" % data_reader.name)
        # Update the generator...if needed
        if self.cfg.EXECUTION.DATAREADER.GENERATOR != data_reader_instance.GENERATOR.__name__:
            data_reader_instance.GENERATOR = locate_class(package="ednaml", subpackage="generators", classpackage=self.cfg.EXECUTION.DATAREADER.GENERATOR)

        self.crawler = self.buildCrawlerInstance(data_reader=data_reader_instance)

        self.buildTrainDataloader(data_reader_instance, self.crawler)
        self.buildTestDataloader(data_reader_instance, self.crawler)

    def buildCrawlerInstance(self, data_reader: DataReader) -> Crawler:
        """Builds a Crawler instance from the data_reader's provided crawler class in `data_reader.CRAWLER`

        Args:
            data_reader (DataReader): A DataReader class

        Returns:
            Crawler: A Crawler instanece for this experiment
        """
        return data_reader.CRAWLER(
            logger=self.logger, **self.cfg.EXECUTION.DATAREADER.CRAWLER_ARGS
        )

    def buildTrainDataloader(self, data_reader: DataReader, crawler_instance: Crawler):
        """Builds a train dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        self.train_generator: ImageGenerator = data_reader.GENERATOR(
            gpus=self.gpus,
            i_shape=self.cfg.TRANSFORMATION.SHAPE,
            normalization_mean=self.cfg.TRANSFORMATION.NORMALIZATION_MEAN,
            normalization_std=self.cfg.TRANSFORMATION.NORMALIZATION_STD,
            normalization_scale=1.0 / self.cfg.TRANSFORMATION.NORMALIZATION_SCALE,
            h_flip=self.cfg.TRANSFORMATION.H_FLIP,
            t_crop=self.cfg.TRANSFORMATION.T_CROP,
            rea=self.cfg.TRANSFORMATION.RANDOM_ERASE,
            rea_value=self.cfg.TRANSFORMATION.RANDOM_ERASE_VALUE,
            **self.cfg.EXECUTION.DATAREADER.GENERATOR_ARGS
        )

        self.train_generator.setup(
            crawler_instance,
            mode="train",
            batch_size=self.cfg.TRANSFORMATION.BATCH_SIZE,
            workers=self.cfg.TRANSFORMATION.WORKERS,
            **self.cfg.EXECUTION.DATAREADER.DATASET_ARGS
        )
        self.logger.info("Generated training data generator")
        self.labelMetadata = self.train_generator.num_entities
        self.logger.info(
            "Running classification model with classes: %s"
            % str(self.labelMetadata.metadata)
        )

    def buildTestDataloader(self, data_reader: DataReader, crawler_instance: Crawler):
        """Builds a test dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        self.test_generator: ImageGenerator = data_reader.GENERATOR(
            gpus=self.gpus,
            i_shape=self.cfg.TRANSFORMATION.SHAPE,
            normalization_mean=self.cfg.TRANSFORMATION.NORMALIZATION_MEAN,
            normalization_std=self.cfg.TRANSFORMATION.NORMALIZATION_STD,
            normalization_scale=1.0 / self.cfg.TRANSFORMATION.NORMALIZATION_SCALE,
            h_flip=0,
            t_crop=False,
            rea=False,
            **self.cfg.EXECUTION.DATAREADER.GENERATOR_ARGS
        )
        self.test_generator.setup(
            crawler_instance,
            mode="test",
            batch_size=self.cfg.TRANSFORMATION.BATCH_SIZE,
            workers=self.cfg.TRANSFORMATION.WORKERS,
            **self.cfg.EXECUTION.DATAREADER.DATASET_ARGS
        )
        self.logger.info("Generated validation data/query generator")

    def buildSaveMetadata(self) -> SaveMetadata:
        """Builds a `SaveMetadata` object containing the model save paths, logger paths, and any other information about saving.
        """
        return SaveMetadata(self.cfg)

    def makeSaveDirectories(self):
        """Creates the save directories for logs, model configuration, metadata, and model save files
        """
        os.makedirs(self.saveMetadata.MODEL_SAVE_FOLDER, exist_ok=True)

    def buildLogger(self, logger: logging.Logger = None) -> logging.Logger:
        """Builds a new logger or adds the correct file and stream handlers to 
        existing logger if it does not already have them. 

        Args:
            logger (logging.Logger, optional): A logger. Defaults to None.

        Returns:
            logging.Logger: A logger with file and stream handlers.
        """
        loggerGiven = True
        if logger is None:
            logger = logging.Logger(self.saveMetadata.MODEL_SAVE_FOLDER)
            loggerGiven = False

        logger_save_path = os.path.join(
            self.saveMetadata.MODEL_SAVE_FOLDER, self.saveMetadata.LOGGER_SAVE_NAME
        )
        # Check for backup logger
        if self.drive_backup:
            backup_logger = os.path.join(
                self.saveMetadata.CHECKPOINT_DIRECTORY,
                self.saveMetadata.LOGGER_SAVE_NAME,
            )
            if os.path.exists(backup_logger):
                print(
                    "Existing log file exists at network backup %s. Will attempt to copy to local directory %s."
                    % (backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER)
                )
                shutil.copy2(backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER)
        if os.path.exists(logger_save_path):
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
                    if handler.baseFilename == os.path.abspath(logger_save_path):
                        filehandler = True

        if not loggerGiven:
            logger.setLevel(logging.DEBUG)

        if not filehandler:
            fh = logging.FileHandler(logger_save_path, mode='a', encoding='utf-8')
            fh.setLevel(self.logLevels[self.verbose])
            formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if not streamhandler:
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
        """Prints the EdnaML configuration
        """
        self.logger.info("*" * 40)
        self.logger.info("")
        self.logger.info("")
        self.logger.info("Using the following configuration:")
        self.logger.info(self.cfg.export())
        self.logger.info("")
        self.logger.info("")
        self.logger.info("*" * 40)
