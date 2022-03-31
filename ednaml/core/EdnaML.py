import os, shutil, logging, glob, re, json
from typing import Dict, List
import warnings
import kaptan
from torchinfo import ModelStatistics
from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators import ImageGenerator
from ednaml.loss.builders import LossBuilder
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.optimizer import BaseOptimizer
from ednaml.optimizer.StandardLossOptimizer import StandardLossOptimizer
from ednaml.loss.builders import ClassificationLossBuilder
from ednaml.trainer.BaseTrainer import BaseTrainer
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
    logLevels = {0:logging.NOTSET,1:logging.ERROR,2:logging.INFO,3:logging.DEBUG}
    labelMetadata: LabelMetadata
    modelStatistics: ModelStatistics
    model: ModelAbstract
    loss_function_array: List[LossBuilder]
    loss_optimizer_array: List[torch.optim.Optimizer]
    optimizer:torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler]
    previous_stop: int
    trainer: BaseTrainer
    crawler: Crawler
    train_generator: ImageGenerator
    test_generator: ImageGenerator

    def __init__(self, config:str="config.yaml", mode:str="train", weights:str=None, logger:logging.Logger=None, verbose:int=2):
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

        self.buildConfig(config)
        self.buildSaveMetadata()
        self.makeSaveDirectories()

        self.gpus = torch.cuda.device_count()
        self.drive_backup = self.cfg.get("SAVE.DRIVE_BACKUP")
        self.previous_stop = 0
        self.epochs = self.cfg.get("EXECUTION.EPOCHS")
        self.skipeval = self.cfg.get("EXECUTION.SKIPEVAL")

        self.step_verbose = self.cfg.get("LOGGING.STEP_VERBOSE")
        self.save_frequency = self.cfg.get("SAVE.SAVE_FREQUENCY")
        self.test_frequency = self.cfg.get("EXECUTION.TEST_FREQUENCY")
        self.fp16 = self.cfg.get("OPTIMIZER.FP16")
        
        self.logger = self.buildLogger(logger=logger)
        
        

    def buildConfig(self, config:str, handler="yaml"):
        """Builds the internal kaptan configuration object

        Args:
            config (str): Path to config file
            handler (str, optional): Handler for kaptan. Defaults to "yaml".
        """
        self.cfg = kaptan.Kaptan(handler=handler)
        self.cfg = self.cfg.import_config(config)


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
            self._download_weights_from_base(self.cfg.get("MODEL.MODEL_BASE"))
        else:
            if self.weights is None:
                warnings.warn("Mode is `test` but weights is `None`. This will cause issues when EdnaML attempts to load weights")


    def _download_weights_from_base(self, model_base: str):
        """Downloads weights from a model_base parameter directly from pytorch's servers.

        Args:
            model_base (str): A supported `model_base`, e.g. resnet18, resnet34. See `utils.web.model_weights`.
        """
        from utils import model_weights
        if model_base in model_weights:
            if os.path.exists(model_weights[model_base][1]):
                pass
            else:
                self.logger.info("Model weights file {} does not exist. Downloading.".format(model_weights[model_base][1]))
                ednaml.utils.web.download(model_weights[model_base][1], model_weights[model_base][0])
            self.pretrained_weights = model_weights[model_base][1]
        else:
            warnings.warn("Model %s is not available. Please choose one of the following: %s if you want to load pretrained weights"%(model_base, str(model_weights.keys())))


    def quickSetup(self):
        """Performs a `quickstart` set-up of EdnaML
        """
        self.setup()
        self.getPreviousStop()  #TODO -- load weights for previous stop outside of trainer...

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
        self.setupTrainer()

    def train(self):
        self.trainer.train(continue_epoch=self.previous_stop)

    def eval(self):
        return self.trainer.evaluate()

    def buildTrainer(self):
        """Builds the EdnaML trainer
        """
        ExecutionTrainer = __import__("trainer", fromlist=["*"])
        ExecutionTrainer = getattr(ExecutionTrainer, self.cfg.get("EXECUTION.TRAINER","ClassificationTrainer"))
        self.logger.info("Loaded {} from {} to build Trainer".format(self.cfg.get("EXECUTION.TRAINER","ClassificationTrainer"), "trainer"))

        self.trainer = ExecutionTrainer( model=self.model, 
                                loss_fn = self.loss_function_array, 
                                optimizer = self.optimizer, 
                                loss_optimizer = self.loss_optimizer_array, 
                                scheduler = self.scheduler, 
                                loss_scheduler = self.loss_scheduler, 
                                train_loader = self.train_generator.dataloader, 
                                test_loader = self.test_generator.dataloader, 
                                epochs = self.epochs, 
                                skipeval = self.skipeval,
                                logger = self.logger, crawler=self.crawler,
                                config = self.cfg,
                                labels = self.labelMetadata)

    def setupTrainer(self):
        """Sets up the EdnaML trainer with logging, saving, gpu, and testing parameters
        """
        self.trainer.setup( step_verbose = self.step_verbose, 
                        save_frequency=self.save_frequency, 
                        test_frequency = self.test_frequency, 
                        save_directory = self.saveMetadata.MODEL_SAVE_FOLDER, 
                        save_backup = self.drive_backup, 
                        backup_directory = self.saveMetadata.CHECKPOINT_DIRECTORY, 
                        gpus=self.gpus,
                        fp16 = self.fp16, 
                        model_save_name = self.saveMetadata.MODEL_SAVE_NAME, 
                        logger_file = self.saveMetadata.LOGGER_SAVE_NAME)

    def getPreviousStop(self):
        """Gets the previous stop, if any, of the trainable model by checking local save directory, as well as a network directory. 
        """
        if self.drive_backup:
            fl_list = glob.glob(os.path.join(self.saveMetadata.CHECKPOINT_DIRECTORY, "*.pth"))
        else:
            fl_list = glob.glob(os.path.join(self.saveMetadata.MODEL_SAVE_FOLDER, "*.pth"))
        _re = re.compile(r'.*epoch([0-9]+)\.pth')
        previous_stop = [int(item[1]) for item in [_re.search(item) for item in fl_list] if item is not None]
        if len(previous_stop) == 0:
            self.previous_stop = 0
            self.logger.info("No previous stop detected. Will start from epoch 0")
        else:
            self.previous_stop = max(previous_stop) + 1
            self.logger.info("Previous stop detected. Will attempt to resume from epoch %i"%self.previous_stop)


    def buildLossScheduler(self):
        """Builds the scheduler for the loss functions, if the functions have learnable parameters and corresponding optimizer.
        """
        self.loss_scheduler = [None]*len(self.loss_optimizer_array)
        for idx, _ in enumerate(self.loss_optimizer_array):
            if self.loss_optimizer_array[idx] is not None:  # In case loss has no differentiable paramters
                try:
                    self.loss_scheduler[idx] = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
                    self.loss_scheduler[idx] = getattr(self.loss_scheduler[idx], self.cfg.get("LOSS_SCHEDULER.LR_SCHEDULER", self.cfg.get("SCHEDULER.LR_SCHEDULER")))
                except (ModuleNotFoundError, AttributeError):
                    loss_scheduler_ = self.cfg.get("LOSS_SCHEDULER.LR_SCHEDULER", self.cfg.get("SCHEDULER.LR_SCHEDULER"))
                    self.loss_scheduler[idx] = __import__("scheduler."+loss_scheduler_, fromlist=[loss_scheduler_])
                    self.loss_scheduler[idx] = getattr(self.loss_scheduler[idx], loss_scheduler_)
                self.loss_scheduler[idx] = self.loss_scheduler[idx](self.loss_optimizer_array[idx], last_epoch = -1, **json.loads(self.cfg.get("LOSS_SCHEDULER.LR_KWARGS", self.cfg.get("SCHEDULER.LR_KWARGS"))))
                self.logger.info("Built loss scheduler")
            else:
                self.loss_scheduler[idx] = None

    def buildScheduler(self):
        """Builds the scheduler for the model
        """
        try:    # We first check if scheduler is part of torch's provided schedulers.
            scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
            scheduler = getattr(scheduler, self.cfg.get("SCHEDULER.LR_SCHEDULER"))
        except (ModuleNotFoundError, AttributeError):   # If it fails, then we try to import from schedulers implemented in scheduler/ folder
            scheduler_ = self.cfg.get("SCHEDULER.LR_SCHEDULER")
            scheduler = __import__("scheduler."+scheduler_, fromlist=[scheduler_])
            scheduler = getattr(scheduler, scheduler_)
        self.scheduler = scheduler(self.optimizer, last_epoch = -1, **json.loads(self.cfg.get("SCHEDULER.LR_KWARGS")))
        self.logger.info("Built scheduler")


    def buildOptimizer(self):
        """Builds the optimizer for the model
        """
        optimizer_builder = __import__("optimizer", fromlist=["*"])
        optimizer_builder:BaseOptimizer = getattr(optimizer_builder, self.cfg.get("EXECUTION.OPTIMIZER_BUILDER", "BaseOptimizer"))
        self.logger.info("Loaded {} from {} to build Optimizer model".format(self.cfg.get("EXECUTION.OPTIMIZER_BUILDER", "OptimizerBuilder"), "optimizer"))

        OPT = optimizer_builder(base_lr=self.cfg.get("OPTIMIZER.BASE_LR"), lr_bias = self.cfg.get("OPTIMIZER.LR_BIAS_FACTOR"), weight_decay=self.cfg.get("OPTIMIZER.WEIGHT_DECAY"), weight_bias=self.cfg.get("OPTIMIZER.WEIGHT_BIAS_FACTOR"), gpus=self.gpus)
        self.optimizer:torch.optim.Optimizer = OPT.build(self.model, self.cfg.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(self.cfg.get("OPTIMIZER.OPTIMIZER_KWARGS")))
        self.logger.info("Built optimizer")
        
    def buildLossOptimizer(self):
        """Builds the Optimizer for loss functions, if the loss functions have learnable parameters (e.g. proxyNCA loss)
        """
        LOSS_OPT = [StandardLossOptimizer(  base_lr=self.cfg.get("LOSS_OPTIMIZER.BASE_LR", self.cfg.get("OPTIMIZER.BASE_LR")), 
                                    lr_bias = self.cfg.get("LOSS_OPTIMIZER.LR_BIAS_FACTOR", self.cfg.get("OPTIMIZER.LR_BIAS_FACTOR")), 
                                    weight_decay= self.cfg.get("LOSS_OPTIMIZER.WEIGHT_DECAY", self.cfg.get("OPTIMIZER.WEIGHT_DECAY")), 
                                    weight_bias= self.cfg.get("LOSS_OPTIMIZER.WEIGHT_BIAS_FACTOR", self.cfg.get("OPTIMIZER.WEIGHT_BIAS_FACTOR")), 
                                    gpus=self.gpus) for _ in self.cfg.get("LOSS")]
        # Note: build returns None if there are no differentiable parameters
        self.loss_optimizer_array = [item.build(loss_builder=self.loss_function_array[idx],
                                        name=self.cfg.get("LOSS_OPTIMIZER.OPTIMIZER_NAME", self.cfg.get("OPTIMIZER.OPTIMIZER_NAME")),
                                        **json.loads(self.cfg.get("LOSS_OPTIMIZER.OPTIMIZER_KWARGS", self.cfg.get("OPTIMIZER.OPTIMIZER_KWARGS"))))
                                        for idx,item in enumerate(LOSS_OPT)]
        self.logger.info("Built loss optimizer")



    def buildLossArray(self):
        """Builds the loss function array using the LOSS list in the provided configuration
        """
        self.loss_function_array = [
            ClassificationLossBuilder(  loss_functions=loss_item["LOSSES"], 
                                        loss_lambda=loss_item["LAMBDAS"], 
                                        loss_kwargs=loss_item["KWARGS"], 
                                        name=loss_item.get("NAME", None),
                                        label=loss_item.get("LABEL", None),
                                        metadata=self.labelMetadata,
                                        **{"logger":self.logger})
            for loss_item in self.cfg.get("LOSS")
        ]
        self.logger.info("Built loss function")


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

        if type(self.cfg.get("MODEL.MODEL_KWARGS")) is dict:  # Compatibility with old configs. TODO fix all old configs.
            model_kwargs_dict = self.cfg.get("MODEL.MODEL_KWARGS")
        elif type(self.cfg.get("MODEL.MODEL_KWARGS")) is None:
            model_kwargs_dict = {}
        else:
            model_kwargs_dict = json.loads(self.cfg.get("MODEL.MODEL_KWARGS"))
        return model_kwargs_dict
    def buildModel(self):
        """Builds an EdnaML model using the configuration. If there are pretrained weights, they are provided through the config to initialize the model.
        """
        model_builder = __import__("models", fromlist=["*"])
        model_builder = getattr(model_builder, self.cfg.get("MODEL.BUILDER", "classification_model_builder"))
        self.logger.info("Loaded {} from {} to build model".format(self.cfg.get("MODEL.BUILDER", "classification_model_builder"), "models"))
        

        model_kwargs = self._covert_model_kwargs()

        # TODO!!!!!!!
        self.model:ModelAbstract  = model_builder(arch = self.cfg.get("MODEL.MODEL_ARCH"), \
                                base=self.cfg.get("MODEL.MODEL_BASE"), \
                                weights=self.pretrained_weights, \
                                metadata = self.labelMetadata, \
                                normalization = self.cfg.get("MODEL.MODEL_NORMALIZATION"), \
                                **model_kwargs)
        self.logger.info("Finished instantiating model with {} architecture".format(self.cfg.get("MODEL.MODEL_ARCH")))

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
                if self.weights != "":   # Load weights if train and starting from a another model base...
                    self.logger.info("Commencing partial model load from {}".format(self.weights))
                    self.model.partial_load(self.weights)
                    self.logger.info("Completed partial model load from {}".format(self.weights))
    def getModelSummary(self):
        """Gets the model summary using `torchinfo` and saves it as a ModelStatistics object
        """
        self.model.cuda()
        self.model_summary = summary(self.model, 
                    input_size=( self.cfg.get("TRANSFORMATION.BATCH_SIZE"), self.cfg.get("TRANSFORMATION.CHANNELS"), *self.cfg.get("TRANSFORMATION.SHAPE")),
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                    depth=3,
                    mode= "train",
                    verbose= 1)
        self.logger.info(str(self.model_summary))





    def buildDataloaders(self):
        """Sets up the datareader classes and builds the train and test dataloaders
        """
        data_reader: DataReader = ednaml.utils.dynamic_import(cfg=self.cfg, 
                                                        module_name="datareaders", 
                                                        import_name="EXECUTION.DATAREADER.DATAREADER", 
                                                        default="VehicleColor")
        
        # data_crawler is now data_reader.CRAWLER
        self.logger.info("Reading data with DataReader %s"%data_reader.name)
        # Update the generator...if needed
        new_generator_class = self.cfg.get("EXECUTION.DATAREADER.GENERATOR", None)
        if new_generator_class is not None:    
            data_reader.GENERATOR = ednaml.utils.dynamic_import(cfg=self.cfg, 
                                            module_name="generators", 
                                            import_name="EXECUTION.DATAREADER.GENERATOR")

        self.crawler = self.buildCrawlerInstance(data_reader=data_reader)

        self.buildTrainDataloader(data_reader, self.crawler)
        self.buildTestDataloader(data_reader, self.crawler)

    def buildCrawlerInstance(self, data_reader: DataReader) -> Crawler:
        """Builds a Crawler instance from the data_reader's provided crawler class in `data_reader.CRAWLER`

        Args:
            data_reader (DataReader): A DataReader class

        Returns:
            Crawler: A Crawler instanece for this experiment
        """
        return data_reader.CRAWLER(logger=self.logger, 
                                    **self.cfg.get("EXECUTION.DATAREADER.CRAWLER_ARGS"))

    def buildTrainDataloader(self, data_reader: DataReader, crawler_instance: Crawler):
        """Builds a train dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        self.train_generator:ImageGenerator = data_reader.GENERATOR(gpus=self.gpus, 
                                i_shape=self.cfg.get("TRANSFORMATION.SHAPE"),
                                normalization_mean=self.cfg.get("TRANSFORMATION.NORMALIZATION_MEAN"), 
                                normalization_std=self.cfg.get("TRANSFORMATION.NORMALIZATION_STD"), 
                                normalization_scale=1./self.cfg.get("TRANSFORMATION.NORMALIZATION_SCALE"),
                                h_flip = self.cfg.get("TRANSFORMATION.H_FLIP"), 
                                t_crop=self.cfg.get("TRANSFORMATION.T_CROP"), 
                                rea=self.cfg.get("TRANSFORMATION.RANDOM_ERASE"), 
                                rea_value=self.cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE"), 
                                **self.cfg.get("EXECUTION.DATAREADER.GENERATOR_ARGS"))

        self.train_generator.setup(crawler_instance, 
                                mode='train',
                                batch_size=self.cfg.get("TRANSFORMATION.BATCH_SIZE"), 
                                workers = self.cfg.get("TRANSFORMATION.WORKERS"), 
                                **self.cfg.get("EXECUTION.DATAREADER.DATASET_ARGS"))
        self.logger.info("Generated training data generator")
        self.labelMetadata = self.train_generator.num_entities
        self.logger.info("Running classification model with classes: %s"%str(self.labelMetadata.metadata))

    def buildTestDataloader(self, data_reader: DataReader, crawler_instance: Crawler):
        """Builds a test dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        self.test_generator:ImageGenerator = data_reader.GENERATOR( gpus=self.gpus, 
                                            i_shape=self.cfg.get("TRANSFORMATION.SHAPE"),
                                            normalization_mean=self.cfg.get("TRANSFORMATION.NORMALIZATION_MEAN"), 
                                            normalization_std=self.cfg.get("TRANSFORMATION.NORMALIZATION_STD"), 
                                            normalization_scale=1./self.cfg.get("TRANSFORMATION.NORMALIZATION_SCALE"),
                                            h_flip = 0, 
                                            t_crop = False, 
                                            rea = False, 
                                            **self.cfg.get("EXECUTION.DATAREADER.GENERATOR_ARGS"))
        self.test_generator.setup(crawler_instance, 
                                mode='test', 
                                batch_size=self.cfg.get("TRANSFORMATION.BATCH_SIZE"), 
                                workers=self.cfg.get("TRANSFORMATION.WORKERS"), 
                                **self.cfg.get("EXECUTION.DATAREADER.DATASET_ARGS"))
        self.logger.info("Generated validation data/query generator")

    def buildSaveMetadata(self):
        """Builds a `SaveMetadata` object containing the model save paths, logger paths, and any other information about saving.
        """
        self.saveMetadata = SaveMetadata(self.cfg)

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

        logger_save_path = os.path.join(self.saveMetadata.MODEL_SAVE_FOLDER, self.saveMetadata.LOGGER_SAVE_NAME)
        # Check for backup logger
        if self.drive_backup:
            backup_logger = os.path.join(self.saveMetadata.CHECKPOINT_DIRECTORY, self.saveMetadata.LOGGER_SAVE_NAME)
            if os.path.exists(backup_logger):
                print("Existing log file exists at network backup %s. Will attempt to copy to local directory %s."%(backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER))
                shutil.copy2(backup_logger, self.saveMetadata.MODEL_SAVE_FOLDER)
        if os.path.exists(logger_save_path):
            print("Log file exists at %s. Will attempt to append there."%logger_save_path)

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
            fh = logging.FileHandler(logger_save_path)
            fh.setLevel(self.logLevels[self.verbose])
            formatter = logging.Formatter('%(asctime)s %(message)s',datefmt="%H:%M:%S")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        if not streamhandler:
            cs = logging.StreamHandler()
            cs.setLevel(self.logLevels[self.verbose])
            cs.setFormatter(logging.Formatter('%(asctime)s %(message)s',datefmt="%H:%M:%S"))
            logger.addHandler(cs)

        return logger

    def log(self, message:str, verbose:int=3):
        """Logs a message. TODO needs to be fixed.

        Args:
            message (str): Message to log
            verbose (int, optional): Logging verbosity. Defaults to 3.
        """
        self.logger.log(self.logLevels[verbose], message)

    def printConfiguration(self):
        """Prints the EdnaML configuration
        """
        self.logger.info("*"*40);self.logger.info("");self.logger.info("")
        self.logger.info("Using the following configuration:")
        self.logger.info(self.cfg.export("yaml", indent=4))
        self.logger.info("");self.logger.info("");self.logger.info("*"*40)