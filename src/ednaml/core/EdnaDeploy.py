from typing import Type
from ednaml.core import EdnaML
import logging

from ednaml.deploy.BaseDeploy import BaseDeploy
from ednaml.utils import locate_class

class EdnaDeploy(EdnaML):
    def __init__(
        self,
        config: str = "",
        mode: str = "test",
        weights: str = None,
        logger: logging.Logger = None,
        verbose: int = 2,
        **kwargs
    ):

        super().__init__(config, mode, weights, logger, verbose, **kwargs)

    

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

    def deploy(self):
        self.deployment.deploy(continue_epoch=self.previous_stop + 1)

    def buildDeployment(self):
        """Builds the EdnaDeploy deployment and sets it up"""
        if self._deploymentClassQueueFlag:
            ExecutionDeployment = self._deploymentClassQueueFlag
        else:
            ExecutionDeployment: Type[BaseDeploy] = locate_class(
                subpackage="deploy", classpackage=self.cfg.DEPLOYMENT.DEPLOYMENT
            )
            self.logger.info(
                "Loaded {} from {} to build Deployment".format(
                    self.cfg.DEPLOYMENT.DEPLOYMENT, "ednaml.deploy"
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