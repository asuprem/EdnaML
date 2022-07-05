from typing import Type
from ednaml.core import EdnaML
import logging
from ednaml.datareaders import DataReader
from ednaml.generators import Generator

from ednaml.deploy.BaseDeploy import BaseDeploy
from ednaml.utils import locate_class

class EdnaDeploy(EdnaML):
    def __init__(
        self,
        config: str = "",
        deploy: str = "",
        mode: str = "test",
        weights: str = None,
        logger: logging.Logger = None,
        verbose: int = 2,
        **kwargs
    ):

        super().__init__([config,deploy], mode, weights, logger, verbose,**kwargs)

    


    def apply(self, **kwargs):
        """Applies the internal configuration for EdnaDeploy"""
        self.printConfiguration()
        self.downloadModelWeights()
        self.setPreviousStop()

        self.buildDataloaders()

        self.buildModel()
        self.loadWeights()
        self.getModelSummary(**kwargs) 

        self.buildDeployment()

        self.resetQueues()


    def buildDataloaders(self):
        """Sets up the datareader classes and builds the train and test dataloaders"""

        data_reader: Type[DataReader] = locate_class(
            package="ednaml",
            subpackage="datareaders",
            classpackage=self.cfg.DEPLOYMENT.DATAREADER.DATAREADER,
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
                self.cfg.DEPLOYMENT.DATAREADER.GENERATOR_ARGS = (
                    self._generatorArgsQueue
                )
        else:
            if (
                self.cfg.DEPLOYMENT.DATAREADER.GENERATOR is not None
            ):
                self.logger.info("Updating GENERATOR using config specification to %s"%self.cfg.DEPLOYMENT.DATAREADER.GENERATOR)
                data_reader_instance.GENERATOR = locate_class(
                    package="ednaml",
                    subpackage="generators",
                    classpackage=self.cfg.DEPLOYMENT.DATAREADER.GENERATOR,
                )

        if self._crawlerClassQueueFlag: #here it checkes whether class flag is set, if it is then replace the build in class with custom class
            self.logger.info("Updating CRAWLER to %s"%self._crawlerClassQueue.__name__)
            data_reader_instance.CRAWLER = self._crawlerClassQueue
            if self._crawlerArgsQueueFlag: #check args also
                self.cfg.DEPLOYMENT.DATAREADER.CRAWLER_ARGS = (
                    self._crawlerArgsQueue
                )

        if self._crawlerInstanceQueueFlag:
            self.crawler = self._crawlerInstanceQueue
        else:
            self.crawler = self._buildCrawlerInstance(
                data_reader=data_reader_instance
            )
        
        # Only need test dataloader...
        self.buildTestDataloader(data_reader_instance, self.crawler)

    def deploy(self):
        self.deployment.deploy(continue_epoch=self.previous_stop + 1)

    def buildDeployment(self):
        """Builds the EdnaDeploy deployment and sets it up"""
        if self._deploymentClassQueueFlag:
            ExecutionDeployment = self._deploymentClassQueueFlag
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
                data_loader=self.test_generator.dataloader, # TODO
                logger=self.logger,
                crawler=self.crawler,
                config=self.cfg,
                labels=self.labelMetadata,
                **self.cfg.DEPLOYMENT.DEPLOYMENT_ARGS
            )
            self.deployment.apply(
                step_verbose=self.cfg.LOGGING.STEP_VERBOSE,
                save_directory=self.saveMetadata.MODEL_SAVE_FOLDER,
                save_backup=self.cfg.SAVE.DRIVE_BACKUP,
                backup_directory=self.saveMetadata.CHECKPOINT_DIRECTORY,
                gpus=self.gpus,
                fp16=self.cfg.EXECUTION.FP16,
                model_save_name=self.saveMetadata.MODEL_SAVE_NAME,
                logger_file=self.saveMetadata.LOGGER_SAVE_NAME,
            )
    
    
    def resetDeploymentQueue(self):
        self._deploymentClassQueue = None
        self._deploymentClassQueueFlag = False
        self._deploymentInstanceQueue = None
        self._deploymentInstanceQueueFlag = False
    
    def addResetQueues(self):
        return [self.resetDeploymentQueue]

    def addDeploymentClass(self, deployment_class: Type[BaseDeploy]):
        self._deploymentClassQueue = deployment_class
        self._deploymentClassQueueFlag = True

    def addDeployment(self, deployment: BaseDeploy):
        self._deploymentInstanceQueue = deployment
        self._deploymentInstanceQueueFlag = True