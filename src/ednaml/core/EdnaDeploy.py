from typing import Type, Union, List
from ednaml.core.EdnaML import EdnaML
import logging
from ednaml.crawlers import Crawler
from ednaml.datareaders import DataReader
from ednaml.generators import Generator

from ednaml.deploy.BaseDeploy import BaseDeploy
from ednaml.storage import StorageManager
from ednaml.utils import locate_class


class EdnaDeploy(EdnaML):
    def __init__(
        self,
        config: str = "",
        deploy: Union[List[str], str] = "",
        mode: str = "test",
        weights: str = None,
        **kwargs
    ):

        super().__init__([config, deploy], mode, weights, **kwargs)

        self.decorator_reference["deployment"] = self.addDeploymentClass
        self.decorator_reference.pop(
            "trainer"
        )  # We do not need trainer in a Deployment
        self.dataloader_mode = kwargs.get("dataloader_mode", self.mode)

    def log(self, msg):
        self.logger.info("[ed]" + msg)

    def debug(self, msg):
        self.logger.debug("[ed]" + msg)

    def apply(self, **kwargs):
        """Applies the internal configuration for EdnaDeploy"""
        self.printConfiguration(**kwargs)
        self.log("[APPLY] Building StorageManager")
        self.buildStorageManager(**kwargs)
        # Build the storage backends that StorageManager can use
        self.log("[APPLY] Adding Storages")
        self.buildStorage(**kwargs)
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
        if not kwargs.get("skip_model", False):
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
        if not kwargs.get("skip_pipeline", False):
            self.log("[APPLY] Building Deployment")
            self.buildDeployment()

        self.resetQueues()

    def buildDataloaders(self):
        """Sets up the datareader classes and builds the train and test dataloaders"""

        data_reader: Type[DataReader] = locate_class(
            package="ednaml",
            subpackage="datareaders",
            classpackage=self.cfg.DATAREADER.DATAREADER,
        )
        data_reader_instance = data_reader()
        self.logger.info("Reading data with DataReader %s" % data_reader_instance.name)
        self.logger.info("Default CRAWLER is %s" % data_reader_instance.CRAWLER)
        self.logger.info("Default DATASET is %s" % data_reader_instance.DATASET)
        self.logger.info("Default GENERATOR is %s" % data_reader_instance.GENERATOR)
        # Update the generator...if needed
        if self._generatorClassQueueFlag:
            self.logger.info(
                "Updating GENERATOR to queued class %s"
                % self._generatorClassQueue.__name__
            )
            data_reader_instance.GENERATOR = self._generatorClassQueue
            if self._generatorArgsQueueFlag:
                self.cfg.DATAREADER.GENERATOR_ARGS = self._generatorArgsQueue
        else:
            if self.cfg.DATAREADER.GENERATOR is not None:
                self.logger.info(
                    "Updating GENERATOR using config specification to %s"
                    % self.cfg.DATAREADER.GENERATOR
                )
                data_reader_instance.GENERATOR = locate_class(
                    package="ednaml",
                    subpackage="generators",
                    classpackage=self.cfg.DATAREADER.GENERATOR,
                )

        if (
            self._crawlerClassQueueFlag
        ):  # here it checkes whether class flag is set, if it is then replace the build in class with custom class
            self.logger.info(
                "Updating CRAWLER to %s" % self._crawlerClassQueue.__name__
            )
            data_reader_instance.CRAWLER = self._crawlerClassQueue
            if self._crawlerArgsQueueFlag:  # check args also
                self.cfg.DATAREADER.CRAWLER_ARGS = self._crawlerArgsQueue

        if self._crawlerInstanceQueueFlag:
            self.crawler = self._crawlerInstanceQueue
        else:
            self.crawler = self._buildCrawlerInstance(data_reader=data_reader_instance)

        # Only need test dataloader...
        self.buildTestDataloader(data_reader_instance, self.crawler)

    def buildTestDataloader(self, data_reader: DataReader, crawler_instance: Crawler):
        """Builds a test dataloader instance given the data_reader class and a crawler instance that has been initialized

        Args:
            data_reader (DataReader): A datareader class
            crawler_instance (Crawler): A crawler instance
        """
        if self._testGeneratorInstanceQueueFlag:
            self.test_generator: Generator = self._testGeneratorInstanceQueue
        else:
            self.logger.info(
                "Generating dataloader `{dataloader}` with `{mode}` mode".format(
                    mode=self.dataloader_mode, dataloader=data_reader.GENERATOR.__name__
                )
            )
            self.test_generator: Type[Generator] = data_reader.GENERATOR(
                logger=self.logger,
                gpus=self.gpus,
                transforms=self.cfg.TEST_TRANSFORMATION,
                mode=self.dataloader_mode,  # TODO convert this to better options: i.e. which mode to use, and which transformations to use, as an option in data_reader, specifically for deployments
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
        self.logger.info("Generated test data/query generator")

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

    def deploy(self, **kwargs):
        self.deployment.deploy(**kwargs)

    def buildDeployment(self):
        """Builds the EdnaDeploy deployment and sets it up"""
        if self._deploymentClassQueueFlag:
            ExecutionDeployment = self._deploymentClassQueue
        else:
            ExecutionDeployment: Type[BaseDeploy] = locate_class(
                subpackage="deploy", classpackage=self.cfg.DEPLOYMENT.DEPLOY
            )
            self.logger.info(
                "Loaded {} from {} to build Deployment".format(
                    self.cfg.DEPLOYMENT.DEPLOY, "ednaml.deploy"
                )
            )

        if self._deploymentInstanceQueueFlag:
            self.deployment = self._deploymentInstanceQueue
        else:
            self.deployment = ExecutionDeployment(
                model=self.model,
                data_loader=self.test_generator.dataloader,  # TODO
                epochs=self.cfg.DEPLOYMENT.EPOCHS,
                logger=self.logger,
                crawler=self.crawler,
                config=self.cfg,
                labels=self.labelMetadata,
                storage=self.storage,
                context=self.context_information,
                **self.cfg.DEPLOYMENT.DEPLOYMENT_ARGS
            )
            # TODO -- change save_backup, backup_directory stuff. These are all in Storage.... We just need the model save name...
            self.deployment.apply(
                step_verbose=self.cfg.LOGGING.STEP_VERBOSE,
                # save_directory=self.saveMetadata.MODEL_SAVE_FOLDER,
                # save_backup=self.cfg.SAVE.DRIVE_BACKUP,
                # save_frequency=self.cfg.SAVE.SAVE_FREQUENCY,
                # backup_directory=self.saveMetadata.CHECKPOINT_DIRECTORY,
                gpus=self.gpus,
                fp16=self.cfg.EXECUTION.FP16,
                storage_manager=self.storageManager,
                log_manager=self.logManager
                # model_save_name=self.saveMetadata.MODEL_SAVE_NAME,
                # logger_file=self.saveMetadata.LOGGER_SAVE_NAME,
            )

    def buildStorageManager(self, **kwargs):  # TODO after I get a handle on the rest...
        self.storageManager = StorageManager(
            logger=self.logger,
            cfg=self.cfg,
            experiment_key=self.experiment_key,
            storage_trigger_mode=kwargs.get("storage_trigger_mode", "loose"),
            storage_manager_mode=kwargs.get(
                "storage_manager_mode", "download_only"
            ),  # Use remote for downloads when provided, but NOT uploads
            storage_mode=kwargs.get("storage_mode", "local"),
            backup_mode=kwargs.get("backup_mode", "hybrid"),
        )

    def resetDeploymentQueue(self):
        self._deploymentClassQueue = None
        self._deploymentClassQueueFlag = False
        self._deploymentInstanceQueue = None
        self._deploymentInstanceQueueFlag = False

    def addResetQueues(self):
        return [self.resetDeploymentQueue]

    def addDeploymentClass(self, deployment_class: Type[BaseDeploy]):
        self.logger.debug("Added deployment class: %s" % deployment_class.__name__)
        self._deploymentClassQueue = deployment_class
        self._deploymentClassQueueFlag = True

    def addDeployment(self, deployment: BaseDeploy):
        self._deploymentInstanceQueue = deployment
        self._deploymentInstanceQueueFlag = True
