import json, logging, os, shutil
from typing import Dict, List, Tuple
import tqdm
import torch
from torch.utils.data import DataLoader
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.core import EdnaMLContextInformation
from ednaml.crawlers import Crawler
from ednaml.logging import LogManager
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.storage import StorageManager
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, KeyMethods, StorageArtifactType
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

    edna_context: EdnaMLContextInformation
    save_frequency: int
    step_save_frequency: int

    def __init__(
        self,
        model: ModelAbstract,
        data_loader: DataLoader,
        epochs: int,
        logger: logging.Logger,
        crawler: Crawler,
        config: EdnaMLConfig,
        labels: LabelMetadata,
        storage: Dict[str, BaseStorage],
        context: EdnaMLContextInformation,
        **kwargs
    ):

        self.model = model
        self.parameter_groups = list(self.model.parameter_groups.keys())
       
        self.data_loader = data_loader
        self.logger = logger
        self.epochs = epochs
        self.global_epoch = 0
        self.global_batch = 0  # Current batch number in the epoch

        self.metadata = {}
        self.labelMetadata = labels
        self.crawler = crawler
        self.config = config
        self.storage = storage

        self.buildMetadata(
            # TODO This is not gonna work with the torchvision wrapper -- ned to fix that; because crawler is not set up for that pattern...?
            crawler=crawler.classes,
            config=json.loads(config.export("json")),
        )

        self.model_is_built = False
        self.edna_context = context

    def buildMetadata(self, **kwargs):
        for keys in kwargs:
            self.metadata[keys] = kwargs.get(keys)

    def apply(
        self,
        step_verbose: int = 5,
        #save_directory: str = "./checkpoint/",
        #save_backup: bool = False,
        #save_frequency: int = 1,
        #step_save_frequency: int = 0,
        #backup_directory: str = None,
        gpus: int = 1,
        fp16: bool = False,
        #model_save_name: str = None,
        #logger_file: str = None,
        storage_manager: StorageManager = None,
        log_manager: LogManager = None,
        storage_mode: str = "loose",    # loose | strict
    ):
        self.step_verbose = step_verbose
        # self.save_directory = save_directory
        # self.save_frequency = save_frequency
        # self.step_save_frequency = step_save_frequency
        # self.backup_directory = None
        # self.model_save_name = model_save_name
        # self.logger_file = logger_file
        # self.save_backup = save_backup
        # if self.save_backup or self.config.SAVE.LOG_BACKUP:
        #     self.backup_directory = backup_directory
        #     os.makedirs(self.backup_directory, exist_ok=True)
        # os.makedirs(self.save_directory, exist_ok=True)
        # self.saveMetadata()


        self.storage_manager = storage_manager
        self.log_manager = log_manager
        self.storage_mode_strict = True if storage_mode == "strict" else False
        self.model_save_name = self.storage_manager.experiment_key.getExperimentName()
        self.gpus = gpus

        if self.gpus != 1:
            self.logger.warning("Multi-gpu or non-gpu not yet fully supported.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.gpus:
            #self.model.cuda() # moves the model into GPU
            self.logger.info("%i GPUs available"%self.gpus)
        self.model.to(self.device)

        self.fp16 = fp16
        # if self.fp16 and self.apex is not None:
        #    self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.output_setup(**self.config.DEPLOYMENT.OUTPUT_ARGS)   # TODO 

    def saveMetadata(self):
        print("NOT saving metadata. saveMetadata() function not set up.")

    def deploy(self, continue_epoch: int = 0, continue_step: int = None, inference = False, ignore_plugins: List[str] = [], execute: bool = True, model_build: bool = None, **kwargs):
        ers_key = self.storage_manager.getLatestERSKey(artifact = StorageArtifactType.MODEL)
        # TODO plugin and model may have different save values since they are not necessarily saved together...
        if continue_epoch is None:  # Use the provided latest key
            self.logger.debug("`continue_epoch` is not provided. Will use latest `ers_key`")
            continue_epoch = ers_key.storage.epoch
            continue_step = ers_key.storage.step
            self.current_ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        else:   # contnue epoch and/or step are provided
            # Check if they are valid. Otherwise, default to provided latest_key
            self.logger.debug("`continue_epoch` is provided. Checking validity in remote and local Storage with artifact MODEL")
            key_exist = self.storage_manager.checkEpoch(
                storage=self.storage, epoch=continue_epoch, artifact=StorageArtifactType.MODEL
            )
            if key_exist is False:
                self.logger.debug("`continue_epoch` value of %i does not exist. Will switch to latest ERSKey."%continue_epoch)
                continue_epoch = None
            else:   # Epoch exists, we now need to check if the step value is valid
                self.logger.debug("`continue_epoch` value of %i is valid. Checking `continue_step`."%continue_epoch)
                if continue_step is not None:
                    key_exist = self.storage_manager.checkStep(
                        storage=self.storage, epoch=continue_epoch, step=continue_step, artifact=StorageArtifactType.MODEL
                    )
                    if key_exist is False:
                        self.logger.debug("`continue_step` value of %i does not exist. Will attempt to find latest step."%continue_step)
                        continue_step = None
                    else:
                        self.logger.debug("`continue_step` value of %i is valid."%continue_step)
                # continue_step is None if not provided, or if the provided value is not valid.
                if continue_step is None:
                    self.logger.debug("`continue_step` is not provided or not valid. Getting latest step saved in Epoch %i in remote and local Storage"%continue_epoch)
                    key = self.storage_manager.getLatestStepOfArtifactWithEpoch(storage=self.storage,epoch=continue_epoch,artifact=StorageArtifactType.MODEL)
                    if key is None:
                        self.logger.debug("No latest step found even though `continue_epoch` is valid. Will switch to latest ERSKey.")
                        continue_step = None
                        continue_epoch = None
                    else:
                        self.logger.debug("Found latest `continue_step` %i"%key.storage.step)
                        continue_step = key.storage.step


            if (continue_epoch is not None) and (continue_step is not None):
                self.logger.info("Using provided epoch/step pair %i/%i."%(continue_epoch,continue_step))
                self.current_ers_key = self.storage_manager.getERSKey(epoch=continue_epoch, step=continue_step)
            else:
                continue_epoch = ers_key.storage.epoch
                continue_step = ers_key.storage.step
                self.current_ers_key = KeyMethods.cloneERSKey(ers_key)
            self.logger.info("Using ERSKey {key}".format(key=self.current_ers_key.printKey()))
        

         # This occurs ONLY if using latestStorageKey and it is empty. TODO this should be 
        # cleaned up for potential bugs when continue_epoch is -1 for other reasons due to 
        # leaks but storageManager is still forced to update StorageKey
        if continue_epoch == -1:
            self.logger.info("Starting deployment from scratch. Setting initial epoch/step to 0/0")
            self.storage_manager.updateStorageKey(self.storage_manager.getNextERSKey())
            self.current_ers_key = self.storage_manager.getLatestERSKey()
            continue_epoch = self.current_ers_key.storage.epoch
            continue_step = self.current_ers_key.storage.step
        else:
            self.logger.info("Starting deployment. with `continue_epoch` %i and `continue_step` %i"%(continue_epoch, continue_step))
            if model_build or (model_build is None and not self.model_is_built):    
                self.logger.info("`model_build` is True or model may not be built. Checking.")
                if continue_epoch or continue_step:
                    if self.edna_context.MODEL_HAS_LOADED_WEIGHTS:
                        self.logger.info("Weights have already been loaded into model. Skipping loading of epoch-specific weights from Epoch %i Step %i"%(continue_epoch, continue_step))
                    else:
                        self.logger.info("Model is empty and `model_build` is set. Attempting loading weights from Epoch %i Step %i"%(continue_epoch, continue_step))
                        response = self.load(continue_epoch, continue_step, ignore_if_error = False, artifact = StorageArtifactType.MODEL)
                        self.logger.info("Model is empty and `model_build` is set. Attempting loading plugins from Epoch %i Step %i"%(continue_epoch, continue_step))
                        plugin_response = self.load(continue_epoch, continue_step, ignore_plugins=ignore_plugins, ignore_if_error = True, artifact = StorageArtifactType.PLUGIN)
                        if not response:
                            raise RuntimeError("Could not load weights at epoch-step %i/%i."%(continue_epoch, continue_step))
                        if not plugin_response:
                            self.logger.info("Could not load plugins at epoch-step %i/%i."%(continue_epoch, continue_step))
                self.model_is_built = True
            else:
                if model_build is not None and not model_build:
                    self.logger.info("Skipping model building and plugin loading due to `model_build=False`")
                elif self.model_is_built:
                    self.logger.info("Skipping model building and plugin loading because model is already built in a prior call to `deploy()`. To force, set the `model_build` flag to True in `ed.deploy`")
                else:
                    self.logger.info("Skipping model building and plugin loading")
        if continue_step == -1:
            raise RuntimeError("`continue_step` is -1 after error checking")


        
        self.logger.info("Logging to:\t%s" % self.log_manager.getLocalLog())
        self.logger.info(
            "Plugins will be saved locally with base name:\t%s"
            % self.storage_manager.getLocalSavePath(self.storage_manager.getERSKey(epoch=0,step=0,artifact_type=StorageArtifactType.PLUGIN))
        )
        self.logger.info("Config will not be saved")
        self.logger.info("Logs will not be saved")
        self.logger.info("Metrics will not be saved")
        self.logger.info("Models will not be saved")
        self.logger.info("Model artifacts will not be saved")

        

        

        if inference:
            self.model.inference()
        else:
            self.model.eval()

        if execute:
            self.logger.info("Setting up plugin hooks. Plugins will fire during:  %s"%self.config.DEPLOYMENT.PLUGIN.HOOKS)
            self.model.set_plugin_hooks(self.config.DEPLOYMENT.PLUGIN.HOOKS)


            self.logger.info("Starting training. with `continue_epoch` %i and `continue_step` %i"%(continue_epoch, continue_step))
            self.logger.info("Logging to:\t%s" % self.log_manager.getLocalLog())

            self.logger.info("Executing deployment for  %i epochs" % self.epochs)
            for epoch in range(self.epochs):
                self.logger.info("Starting epoch %i"%self.global_epoch)
                self.model.pre_epoch_hook(epoch=epoch)
                self.data_step()
                self.model.post_epoch_hook(epoch=epoch)
                
                
                
                
                self.check_epoch_save(self.global_epoch)
                self.logger.info(
                    "{0} Completed epoch {1} {2}".format(
                        "*" * 10, self.global_epoch, "*" * 10
                    )
                )
                
                self.global_epoch = epoch + 1
                
                self.logger.info("Executing end of epoch steps")
                self.end_of_epoch(epoch=epoch)
            self.end_of_deployment()
            self.logger.info("Completed deployment task.")
        else:
            self.logger.info("Skipping execution of deployment task due to `execute = False`.")

    def data_step(self):   
        with torch.no_grad():
            for batch in tqdm.tqdm(
                self.data_loader, total=len(self.data_loader), leave=False
            ):    
                batch = self.move_to_device(batch)
                feature_logits, features, secondary_outputs = self.deploy_step(batch)

                self.output_step(feature_logits, features, secondary_outputs)
                # Log Metrics here and inside the model TODO
                
                self.global_batch += 1
            # No saving state in the middle of a deployment...

    def move_to_device(self, batch) -> Tuple[torch.Tensor]:
        return (item.to(self.device) for item in batch)
        
    def deploy_step(self, batch):   # USER IMPLEMENTS
        data, labels = batch    # TODO move plugins here to allow labels as well!!!!!!!!
        feature_logits, features, secondary_outputs = self.model(data)

        return feature_logits, features, secondary_outputs

    def end_of_epoch(self, epoch: int):
        self.logger.warn("No end of epochs steps are performed")
    
    def end_of_deployment(self):
        self.logger.warn("No end of deployment steps are performed")

    def output_setup(self, **kwargs): # USER IMPLEMENTS; kwargs from config.DEPLOYMENT.OUTPUT_ARGS
        self.logger.warn("No output setup is performed")

    def output_step(self, logits, features, secondary): # USER IMPLEMENTS, ALSO, NEED SOME STEP LOGGING...????????
        if self.global_batch % self.config.LOGGING.STEP_VERBOSE == 0:
            self.logger.warn("No output is generated at step %i"%self.global_batch)



    def check_epoch_save(self, epoch):  # TODO off by one errors
        self.logger.debug("Checking epoch save status at Epoch %i"%epoch)
        if self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.MODEL):
            # For gradient accumulation
            self.save(artifact=StorageArtifactType.MODEL, save_epoch=epoch, save_step = 0)
        if self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.PLUGIN):
            self.save(artifact=StorageArtifactType.PLUGIN, save_epoch=epoch, save_step = 0)
        if self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.LOG):
            self.save(artifact=StorageArtifactType.LOG, save_epoch=epoch, save_step = 0)
        if self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.METRIC):
            self.save(artifact=StorageArtifactType.METRIC, save_epoch=epoch, save_step = 0)
        if self.storage_manager.getUploadTriggerForStep(epoch, StorageArtifactType.CONFIG):
            self.save(artifact=StorageArtifactType.CONFIG, save_epoch=epoch, save_step = 0)


    # load is called by self.train() when being initialized...
    def load(self, load_epoch: int = None, load_step : int = None, artifact: StorageArtifactType = StorageArtifactType.MODEL, ignore_if_error: bool = False, ignore_plugins: List[str] = []):
        """_summary_

        Args:
            load_epoch (_type_): _description_
            load_step (int, optional): _description_. Defaults to 0.
        """
        self.logger.info(
            "Loading saved state of %s from epoch %i / step %i"
            % (artifact.value, load_epoch, load_step)
        )

        # TODO what happens when there is a local copy as well as a remote copy -- we do not need to download from remote...
        # Have StorageManager handle this gracefully...
        if artifact == StorageArtifactType.MODEL:
            model_ers_key: ERSKey = self.storage_manager.getERSKey(epoch = load_epoch, step = load_step, artifact_type=artifact)
            response = self.storage_manager.download(
                ers_key=model_ers_key, storage_dict=self.storage
            ) 

            if response:
                model_local_storage_savepath = self.storage_manager.getLocalSavePath(ers_key=model_ers_key)
                # TODO replace with ModelAbstract's own loader?????
                if self.gpus == 0:
                    self.model.load_state_dict(torch.load(model_local_storage_savepath, map_location=self.device))
                else:
                    self.model.load_state_dict(torch.load(model_local_storage_savepath))
                self.logger.info(
                    "Finished loading model state_dict from %s" % model_local_storage_savepath
                )
            else:
                if not ignore_if_error:
                    raise FileNotFoundError("Could not download artifact %s from epoch %i ? step %i"%(artifact.value, load_epoch, load_step))
                return False

        elif artifact == StorageArtifactType.ARTIFACT:
            raise NotImplementedError("MODEL_ARTIFACT is loaded with MODEL")
        elif artifact == StorageArtifactType.CONFIG:
            raise NotImplementedError("CONFIG downloads should not be managed inside Deploy")
        elif artifact == StorageArtifactType.LOG:
            raise NotImplementedError("LOG downloads should not be managed inside Deploy")
        elif artifact == StorageArtifactType.PLUGIN:
            plugin_ers_key: ERSKey = self.storage_manager.getERSKey(epoch = load_epoch, step = load_step, artifact_type=artifact)
            response = self.storage_manager.download(
                ers_key=plugin_ers_key, storage_dict=self.storage
            ) 
            if response:
                plugin_local_savepath = self.storage_manager.getLocalSavePath(ers_key=plugin_ers_key)
                self.model.loadPlugins(plugin_local_savepath, ignore_plugins=ignore_plugins)
                self.logger.info(
                    "Finished loading plugin state_dict from %s" % plugin_ers_key
                )
            else:
                if not ignore_if_error:
                    raise FileNotFoundError("Could not download artifact %s from epoch %i ? step %i"%(artifact.value, load_epoch, load_step))
                return False




        elif artifact == StorageArtifactType.METRIC:
            raise NotImplementedError("METRIC downloads should not be managed inside Deploy")
        else:
            raise ValueError("Unexpected value for artifact type %s"%artifact.value)
        
        return True
       

    def save(self, save_epoch: int = None, save_step : int = None, artifact: StorageArtifactType = StorageArtifactType.MODEL):
        if save_epoch is None:
            save_epoch = self.global_epoch
        if save_step is None:
            save_step = self.global_batch
        if self.storage_manager.storage_mode == "local":
            self.logger.debug("Attempting upload of artifact `%s` at epoch %i / step %i"%(artifact.value, save_epoch, save_step))
            # For whatever the artifact is, we will first perform a local save,
            # then perform a backup IF backup_perform is true...
            if artifact == StorageArtifactType.MODEL:
                self.logger.debug("Not saving model in Deployment")
            elif artifact == StorageArtifactType.ARTIFACT:
                raise ValueError("Cannot save MODEL_ARTIFACT by itself. Call `save()` for MODEL to save MODEL_ARTIFACT.")
            elif artifact == StorageArtifactType.PLUGIN:
                
                plugin_ers_key = self.storage_manager.getERSKey(epoch=save_epoch, step=save_step, artifact_type=artifact)
                plugin_local_path = self.storage_manager.getLocalSavePath(ers_key=plugin_ers_key)

                plugin_save = self.model.savePlugins()

                if len(plugin_save) > 0:
                    torch.save(plugin_save, plugin_local_path)
                    self.logger.debug("Saved the following plugins: %s"%str(plugin_save.keys()))

                    if self.storage_manager.performBackup(artifact):    # TODO under strict/loose modes, storageManager can take care of all of these. 
                        self.storage_manager.upload(
                            self.storage,
                            plugin_ers_key
                        )
                else:
                    self.logger.info("No plugins to save")
            elif artifact == StorageArtifactType.LOG:
                self.logger.debug("Not saving logs in Deployment") 
            elif artifact == StorageArtifactType.CONFIG:
                self.logger.debug("Not saving config in Deployment") 
            elif artifact == StorageArtifactType.METRIC:    # TODO skip for now
                # MetricManager writes to one-file-per-metric, or to remote server (or both).
                # We call MetricManager once in a while to dump in-memory data to disk, or to batch send them to backend server 
                metrics_ers_key = self.storage_manager.getERSKey(epoch=save_epoch, step=save_step, artifact_type=StorageArtifactType.METRIC)
                #metrics_filename = self.storage_manager.getLocalSavePath(metrics_ers_key)
                # NOTE: There can be multiple metrics files, with pattern: [metric_name].metrics.json
                if self.storage_manager.performBackup(artifact):
                    self.storage_manager.upload(
                        self.storage,
                        metrics_ers_key
                    )
                    #self.storage[self.storage_manager.getStorageNameForArtifact(artifact)].uploadMetric(source_file_name=metrics_filename,ers_key=metrics_ers_key)
            else:
                raise ValueError("Unexpected value for artifact type %s"%artifact.value)

        else:   # TODO could this be more graceful / elegant
            self.logger.info("Not uploading artifact `%s` due to empty storage"%(artifact.value))

       
        