import json, logging, os, shutil
from typing import Dict, List
import tqdm
import torch
from torch.utils.data import DataLoader
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.crawlers import Crawler
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils.LabelMetadata import LabelMetadata


class BaseDeploy:
    """Base deployment class
    """
    model: ModelAbstract
    data_loader: DataLoader

    global_batch: int
    metadata: Dict[str, str]
    labelMetadata: LabelMetadata
    logger: logging.Logger

    def __init__(
        self,
        model: ModelAbstract,
        data_loader: DataLoader,
        logger: logging.Logger,
        crawler: Crawler,
        config: EdnaMLConfig,
        labels: LabelMetadata,
        **kwargs
    ):

        self.model = model
        self.parameter_groups = list(self.model.parameter_groups.keys())
       
        self.data_loader = data_loader
        self.logger = logger

        self.global_batch = 0  # Current batch number in the epoch

        self.metadata = {}
        self.labelMetadata = labels
        self.crawler = crawler
        self.config = config

        self.buildMetadata(
            # TODO This is not gonna work with the torchvision wrapper -- ned to fix that; because crawler is not set up for that pattern...?
            crawler=crawler.classes,
            config=json.loads(config.export("json")),
        )

    def buildMetadata(self, **kwargs):
        for keys in kwargs:
            self.metadata[keys] = kwargs.get(keys)

    def apply(
        self,
        step_verbose: int = 5,
        save_directory: str = "./checkpoint/",
        save_backup: bool = False,
        backup_directory: str = None,
        gpus: int = 1,
        fp16: bool = False,
        model_save_name: str = None,
        logger_file: str = None,
    ):
        self.step_verbose = step_verbose
        self.save_directory = save_directory
        self.backup_directory = None
        self.model_save_name = model_save_name
        self.logger_file = logger_file
        self.save_backup = save_backup
        if self.save_backup or self.config.SAVE.LOG_BACKUP:
            self.backup_directory = backup_directory
            os.makedirs(self.backup_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)
        self.saveMetadata()

        self.gpus = gpus

        if self.gpus != 1:
            raise NotImplementedError()

        self.model.cuda() # moves the model into GPU

        self.fp16 = fp16
        # if self.fp16 and self.apex is not None:
        #    self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.output_setup(**self.config.DEPLOYMENT.OUTPUT_ARGS)   # TODO 
    def saveMetadata(self):
        print("NOT saving metadata. saveMetadata() function not set up.")

    def deploy(self, continue_epoch=0, inference = False):
        self.logger.info("Starting deployment")
        self.logger.info("Logging to:\t%s" % self.logger_file)
        if self.config.SAVE.LOG_BACKUP:
            self.logger.info(
                "Logs will be backed up to drive directory:\t%s"
                % self.backup_directory
            )
        
        self.logger.info("Loading model from saved epoch %i" % continue_epoch)
        if continue_epoch > 0:
            load_epoch = continue_epoch - 1
            self.load(load_epoch)

        if inference:
            self.model.inference()
        else:
            self.model.eval()
        self.data_step()

        self.logger.info("Completed deployment task.")

    def data_step(self):   
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.test_loader, total=len(self.test_loader), leave=False
            ):    
                feature_logits, features, secondary_outputs = self.deploy_step(batch)

                self.output_step(feature_logits, features, secondary_outputs)
                # Log Metrics here and inside the model TODO
                self.global_batch += 1


    def deploy_step(self, batch):   # USER IMPLEMENTS
        batch = tuple(item.cuda() for item in batch)
        data, labels = batch
        feature_logits, features, secondary_outputs = self.model(data)

        return feature_logits, features, secondary_outputs



    def output_setup(self, **kwargs): # USER IMPLEMENTS; kwargs from config.DEPLOYMENT.OUTPUT_ARGS
        self.logger.info("Warning: No output setup is performed")

    def output_step(self, logits, features, secondary): # USER IMPLEMENTS, ALSO, NEED SOME STEP LOGGING...????????
        if self.global_batch % self.config.LOGGING.STEP_VERBOSE == 0:
            self.logger.info("Warning: No output is generated at step %i"%self.global_batch)


