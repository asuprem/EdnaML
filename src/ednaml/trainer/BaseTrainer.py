import  logging, shutil
from logging import Logger
from typing import Dict, List, Tuple, Type, Any, Union
import torch
from torch.utils.data import DataLoader
from ednaml.core import EdnaMLContextInformation
from ednaml.logging import LogManager
import ednaml.loss.builders
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.crawlers import Crawler
from ednaml.models.ModelAbstract import ModelAbstract
from ednaml.utils import ERSKey, KeyMethods, StorageArtifactType, StorageKey
from ednaml.utils.LabelMetadata import LabelMetadata
from ednaml.storage import BaseStorage, StorageManager
from ednaml.utils import locate_class
from ednaml.metrics import BaseMetric, MetricsManager



class BaseTrainer:
    model: ModelAbstract
    loss_fn: Dict[str, ednaml.loss.builders.LossBuilder]  # output-name: lossBuilder
    optimizer: Dict[str, torch.optim.Optimizer]
    loss_optimizer = Dict[str, List[torch.optim.Optimizer]]
    scheduler: Dict[str, torch.optim.lr_scheduler._LRScheduler]
    loss_scheduler: Dict[str, List[torch.optim.lr_scheduler._LRScheduler]]

    skipeval: bool
    train_loader: DataLoader
    test_loader: DataLoader

    epochs: int
    logger: Logger
    global_batch: int
    global_epoch: int
    num_losses: int
    losses: Dict[str, List[int]]
    metadata: Dict[str, str]
    labelMetadata: LabelMetadata
    logger: logging.Logger
    storage: Dict[str, BaseStorage]
    storage_manager: StorageManager
    metrics_manager: MetricsManager
    log_manager: LogManager
    storage_mode_strict: bool

    save_frequency: int
    step_save_frequency: int

    edna_context: EdnaMLContextInformation
    saveFlag_epoch: int
    saveFlag_step: int
    saveFlag_localonly: bool

    current_ers_key: ERSKey
    model_metrics: Dict[str, BaseMetric]
    artifact_metrics: Dict[str, BaseMetric]
    model_params: Dict[str, Any]
    artifact_params: Dict[str, Any]
    model_metrics_enable: bool
    artifact_metrics_enable: bool
    model_always_metrics: List[str]
    model_immediate_metrics: List[str]
    model_step_metrics: List[str]
    model_batch_metrics: List[str]
    artifact_always_metrics: List[str]
    artifact_immediate_metrics: List[str]
    artifact_step_metrics: List[str]
    artifact_batch_metrics: List[str]

    def __init__(
        self,
        model: ModelAbstract,
        loss_fn: List[ednaml.loss.builders.LossBuilder],
        optimizer: List[torch.optim.Optimizer],
        loss_optimizer: List[torch.optim.Optimizer],
        scheduler: List[torch.optim.lr_scheduler._LRScheduler],
        loss_scheduler: List[torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        skipeval: bool,
        logger: Logger,
        crawler: Crawler,
        config: EdnaMLConfig,
        labels: LabelMetadata,
        storage: Dict[str, BaseStorage],
        context: EdnaMLContextInformation,
        **kwargs
    ):
        self.model_params = {}
        self.artifact_params = {}
        self.model = model
        self.parameter_groups = list(self.model.parameter_groups.keys())
        self.loss_fn_order = {
            idx: lossbuilder.loss_labelname for idx, lossbuilder in enumerate(loss_fn)
        }
        self.loss_fn = {
            lossbuilder.loss_labelname: lossbuilder for lossbuilder in loss_fn
        }
        self.num_losses = len(self.loss_fn)
        self.losses = {lossname: [] for lossname in self.loss_fn}
        self.loss_optimizer = {
            self.loss_fn_order[idx]: loss_optimizer_content
            for idx, loss_optimizer_content in enumerate(loss_optimizer)
        }
        if type(loss_scheduler) is list:
            self.loss_scheduler = {
                self.loss_fn_order[idx]: loss_scheduler_content
                for idx, loss_scheduler_content in enumerate(loss_scheduler)
            }
        else:
            self.loss_scheduler = {
                self.loss_fn_order[idx]: loss_scheduler
                for idx in range(self.num_losses)
            }

        self.optimizer = {
            self.parameter_groups[idx]: optimizer_item
            for idx, optimizer_item in enumerate(optimizer)
        }
        self.scheduler = {
            self.parameter_groups[idx]: scheduler_item
            for idx, scheduler_item in enumerate(scheduler)
        }
        self.skipeval = skipeval
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.logger = logger

        self.global_batch = 0  # Current batch number in the epoch
        self.global_epoch = 0
        self.saveFlag_epoch = 0
        self.saveFlag_step = 0
        self.saveFlag_localonly = True

        self.metadata = {}
        self.labelMetadata = labels
        self.crawler: Crawler = crawler
        self.config: EdnaMLConfig = config
        self.storage = storage

        self.model_metrics = {}
        self.model_always_metrics = []
        self.model_immediate_metrics = []
        self.model_step_metrics = []
        self.model_batch_metrics = []
        self.model_metrics_enable = False

        self.artifact_metrics = {}
        self.artifact_always_metrics = []
        self.artifact_immediate_metrics = []
        self.artifact_step_metrics = []
        self.artifact_batch_metrics = []
        self.artifact_metrics_enable = False

        
        self.accumulation_steps = kwargs.get("accumulation_steps")
        self.accumulation_count = 0
        self.evaluateFlag = False
        self.saveFlag = False
        self.edna_context = context
        self.model_params["model"] = self.model
        


    def addModelMetrics(self, metrics_list: List[BaseMetric], epoch, step):
        self.model_immediate_metrics: List[BaseMetric] = []
        if len(metrics_list) > 0:
            # Check for metrics that are executed only once, e.g. now
            for metric in metrics_list:
                self.model_metrics[metric.metric_name] = metric
                if metric.metric_trigger == "once":
                    # Metric is triggered only once.
                    # We will trigger it at the end of addMetrics()

                    self.model_immediate_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "always":
                    # Metric is `always` triggered. Always has different meanings for each Manager. We will add it to our bookkeeping
                    self.model_always_metrics.append(metric.metric_name)

                elif metric.metric_trigger == "step":
                    self.model_step_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "batch":
                    self.model_batch_metrics.append(metric.metric_name)
                else:
                    raise ValueError("metric_trigger %s is not supported"%metric.metric_trigger)
            self.model_metrics_enable = True
        else:
            self.model_metrics_enable = False

        if self.model_metrics_enable:
            # TODO should we get the next key or the latest key?
            for metric_name in self.model_immediate_metrics:
                self.model_metrics[metric_name].update(epoch, step, params=self.params)

            # Unused for now
            if len(self.model_always_metrics):
                self.model_always_enable = True # Enable always metrics
            if len(self.model_step_metrics):
                self.model_step_enable = True # Enable step metrics
            if len(self.model_batch_metrics):
                self.model_batch_enable = True # Enable batch metrics

    def addArtifactMetrics(self, metrics_list: List[BaseMetric], epoch, step):
        self.artifact_immediate_metrics: List[BaseMetric] = []
        if len(metrics_list) > 0:
            # Check for metrics that are executed only once, e.g. now
            for metric in metrics_list:
                self.artifact_metrics[metric.metric_name] = metric
                if metric.metric_trigger == "once":
                    # Metric is triggered only once.
                    # We will trigger it at the end of addMetrics()

                    self.artifact_immediate_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "always":
                    # Metric is `always` triggered. Always has different meanings for each Manager. We will add it to our bookkeeping
                    self.artifact_always_metrics.append(metric.metric_name)

                elif metric.metric_trigger == "step":
                    self.artifact_step_metrics.append(metric.metric_name)
                elif metric.metric_trigger == "batch":
                    self.artifact_batch_metrics.append(metric.metric_name)
                else:
                    raise ValueError("metric_trigger %s is not supported"%metric.metric_trigger)
            self.artifact_metrics_enable = True
        else:
            self.artifact_metrics_enable = False

        if self.artifact_metrics_enable:
            # TODO should we get the next key or the latest key?
            for metric_name in self.artifact_immediate_metrics:
                self.artifact_metrics[metric_name].update(epoch, step, params=self.params)

            # Unused for now
            if len(self.artifact_always_metrics):
                self.artifact_always_enable = True # Enable always metrics
            if len(self.artifact_step_metrics):
                self.artifact_step_enable = True # Enable step metrics
            if len(self.artifact_batch_metrics):
                self.artifact_batch_enable = True # Enable batch metrics

    
    # For now, we ignore always metrics
    def updateStepMetrics(self, epoch, step):
        for metric_name in self.model_step_metrics:
            self.model_metrics[metric_name].update(epoch=epoch, step=step, params=self.model_params)
        for metric_name in self.artifact_step_metrics:
            self.artifact_metrics[metric_name].update(epoch=epoch, step=step, params=self.artifact_params)


        self.log_manager.updateStepMetrics(epoch, step)
        self.metrics_manager.updateStepMetrics(epoch, step)
        self.config.updateStepMetrics(epoch, step)
        # Code metrics
        # Plugin Metrics
        # Storage metrics????

    def updateBatchMetrics(self, epoch, step):
        for metric_name in self.model_batch_metrics:
            self.model_metrics[metric_name].update(epoch=epoch, step=step, params=self.model_params)
        for metric_name in self.artifact_batch_metrics:
            self.artifact_metrics[metric_name].update(epoch=epoch, step=step, params=self.artifact_params)


        self.log_manager.updateBatchMetrics(epoch, step)
        self.metrics_manager.updateBatchMetrics(epoch, step)
        self.config.updateBatchMetrics(epoch, step)
        

    def buildMetadata(self, **kwargs):
        for keys in kwargs:
            self.metadata[keys] = kwargs.get(keys)

    def apply(
        self,
        step_verbose: int = 5,
        save_frequency: int = 0,
        step_save_frequency: int = 0,
        test_frequency: int = 5,
        storage_manager: StorageManager = None,
        log_manager: LogManager = None,
        metrics_manager: MetricsManager = None,
        storage_mode: str = "loose",  # loose | strict
        gpus: int = 1,
        fp16: bool = False,
    ):
        self.step_verbose = step_verbose
        self.save_frequency = save_frequency
        self.step_save_frequency = step_save_frequency
        self.test_frequency = test_frequency
        self.storage_manager = storage_manager
        self.metrics_manager = metrics_manager
        self.log_manager = log_manager
        self.model_save_name = self.storage_manager.experiment_key.getExperimentName()
        self.gpus = gpus

        if self.gpus != 1:
            self.logger.warning("Multi-gpu or non-gpu not yet fully supported.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.gpus:
            self.logger.info("%i GPUs available" % self.gpus)
        self.model.to(self.device)

        self.fp16 = fp16
        # if self.fp16 and self.apex is not None:
        #    self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')

        # Update the Metrics Metadata with LabelMetadata
        self.metrics_manager.updateMetadata(label_metadata = self.labelMetadata)


    def saveMetadata(self):
        print("NOT saving metadata. saveMetadata() function not set up.")

    def save(
        self,
        save_epoch: int = None,
        save_step: int = None,
        artifact: StorageArtifactType = StorageArtifactType.MODEL,
        local_only = False
    ):
        if save_epoch is None:
            save_epoch = self.global_epoch
        if save_step is None:
            save_step = self.global_batch
        if self.storage_manager.storage_mode == "local":
            self.logger.info(
                "Attempting save of artifact `%s` at epoch %i / step %i, in `local_only=%s` mode."
                % (artifact.value, save_epoch, save_step, str(local_only))
            )
            # For whatever the artifact is, we will first perform a local save,
            # then perform a backup IF backup_perform is true...
            if artifact == StorageArtifactType.MODEL:
                self.saveModel(epoch=save_epoch, step=save_step, local_only = local_only)
                self.saveArtifact(epoch=save_epoch, step=save_step, local_only = local_only)
            elif artifact == StorageArtifactType.ARTIFACT:
                raise ValueError(
                    "Cannot save MODEL_ARTIFACT by itself. Call `save()` for MODEL to save MODEL_ARTIFACT."
                )
            elif artifact == StorageArtifactType.PLUGIN:
                self.savePlugin(epoch=save_epoch, step=save_step, local_only = local_only)
            elif artifact == StorageArtifactType.LOG:
                self.saveLog(epoch=save_epoch, step=save_step, local_only = local_only)
            elif artifact == StorageArtifactType.CONFIG:
                self.saveConfig(epoch=save_epoch, step=save_step, local_only = local_only)
            elif artifact == StorageArtifactType.METRIC:
                self.saveMetrics(epoch=save_epoch, step=save_step, local_only = local_only)
            else:
                raise ValueError(
                    "Unexpected value for artifact type %s" % artifact.value
                )
        else:  # TODO could this be more graceful / elegant
            self.logger.info(
                "Not saving or uploading artifact `%s` due to empty storage" % (artifact.value)
            )

        # TODO need to check if ModelAbstract contains the plugins that came with it...
        # if not, we will need to save plugins in a plugins artifact.
        # And even if they did, we need to split up model and plugins, since they are not strictly part of the model.

    # load is called by self.train() when being initialized...

    def saveModel(self, epoch: int, step: int, local_only: bool = False):
        model_ers_key: ERSKey = self.storage_manager.getERSKey(
            epoch=epoch, step=step, artifact_type=StorageArtifactType.MODEL
        )
        model_local_storage_savepath = self.storage_manager.getLocalSavePath(
            ers_key=model_ers_key
        )

        torch.save(
            self.model.state_dict(),
            model_local_storage_savepath,
        )
        if not local_only:
            if self.storage_manager.performBackup(
                StorageArtifactType.MODEL
            ):  # TODO under strict/loose modes, storageManager can take care of all of these.
                self.storage_manager.upload(self.storage, model_ers_key)

    def saveArtifact(self, epoch: int, step: int, local_only: bool = False):
        save_dict = {}
        save_dict["optimizer"] = {
            pgn: self.optimizer[pgn].state_dict() for pgn in self.parameter_groups
        }
        save_dict["scheduler"] = {
            pgn: self.scheduler[pgn].state_dict() for pgn in self.parameter_groups
        }
        save_dict["loss_fn"] = {
            lossname: self.loss_fn[lossname].state_dict() for lossname in self.loss_fn
        }
        save_dict["loss_optimizer"] = {
            lossname: (
                self.loss_optimizer[lossname].state_dict()
                if self.loss_optimizer[lossname] is not None
                else None
            )
            for lossname in self.loss_optimizer
        }
        save_dict["loss_scheduler"] = {
            lossname: (
                self.loss_scheduler[lossname].state_dict()
                if self.loss_scheduler[lossname] is not None
                else None
            )
            for lossname in self.loss_scheduler
        }

        artifact_ers_key = self.storage_manager.getERSKey(
            epoch=epoch, step=step, artifact_type=StorageArtifactType.ARTIFACT
        )
        artifact_local_storage_savepath = self.storage_manager.getLocalSavePath(
            artifact_ers_key
        )
        torch.save(save_dict, artifact_local_storage_savepath)
        if not local_only:
            if self.storage_manager.performBackup(
                StorageArtifactType.ARTIFACT
            ):  # TODO under strict/loose modes, storageManager can take care of all of these.
                self.storage_manager.upload(self.storage, artifact_ers_key)

    def saveLog(self, epoch: int = None, step: int = None, local_only: bool = False):
        self._generic_save(epoch=epoch,step=step,local_only=local_only,manager=self.log_manager,artifact=StorageArtifactType.LOG)

    def saveMetrics(self, epoch: int, step: int, local_only: bool = False):
        self._generic_save(epoch=epoch,step=step,local_only=local_only,manager=self.metrics_manager,artifact=StorageArtifactType.METRIC)

    def savePlugin(self, epoch: int, step: int, local_only: bool = False):
        self.logger.debug("Not saving model plugins")  # TODO

    def saveConfig(self, epoch: int, step: int, local_only: bool = False):
        self._generic_save(epoch=epoch,step=step,local_only=local_only,manager=self.config,artifact=StorageArtifactType.CONFIG)



    def _generic_save(self, epoch: int, step: int, manager: Union[MetricsManager,LogManager,EdnaMLConfig], local_only: bool, artifact: StorageArtifactType):
        #      Flush the Manager
        #      Request Manager to save file
        #      Request file from Manager
        #      If file path is NOT the same as current ERSKey, copy file to local ERSKey
        #      Ask StorageManager to upload the file if backup is allowed.
        if epoch is None:
            epoch = self.global_epoch
        if step is None:
            step = self.global_batch
        self.logger.info("Flushing {artifact}".format(artifact=artifact.value))    
        # manager --> LogManager, MetricsManager, EdnaMLConfig, CodeManager (future)
        
        manager.flush()     # Dumps running processes
        # We will deal with serialization later...
        manager.save(epoch, step)   # Generate file. Some managers generate file during flushing, and not during save()
        # TODO can be serialized...

        local_artifact_file = manager.getLocalFile()
        artifact_ers_key = self.storage_manager.getERSKey(
            epoch=epoch, step=step, artifact_type=artifact
        )
        storage_artifact_file = self.storage_manager.getLocalSavePath(ers_key=artifact_ers_key)

        if local_artifact_file != storage_artifact_file and local_artifact_file != "":
            self.logger.info(
                "Transferring compatible {artifact} file at {metricspath} to StorageManager path at {stpath}".format(
                    artifact=artifact.value, metricspath=local_artifact_file, stpath=storage_artifact_file
                )
            )
            shutil.copy2(local_artifact_file, storage_artifact_file)

        if not local_only:
            self.logger.info(
                "Can upload {artifact} with key {metrics_key}".format(
                    artifact=artifact.value,
                    metrics_key=artifact_ers_key.printKey()
                )
            )
            if self.storage_manager.performBackup(artifact):
                self.logger.info(
                    "Attempting upload of {artifact} at StorageManager path {stpath} with key {metrics_key}".format(
                        artifact=artifact.value,
                        stpath=storage_artifact_file,
                        metrics_key=artifact_ers_key.printKey()
                    )
                )
                self.storage_manager.upload(self.storage, artifact_ers_key)
    

    def saveCode(self, epoch: int, step: int, local_only: bool = False):
        pass

    def load(
        self,
        load_epoch: int = None,
        load_step: int = None,
        artifact: StorageArtifactType = StorageArtifactType.MODEL,
        ignore_if_error: bool = False,
    ):
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
            model_ers_key: ERSKey = self.storage_manager.getERSKey(
                epoch=load_epoch, step=load_step, artifact_type=artifact
            )
            response = self.storage_manager.download(
                ers_key=model_ers_key, storage_dict=self.storage
            )
            artifact_ers_key: ERSKey = self.storage_manager.getERSKey(
                epoch=load_epoch,
                step=load_step,
                artifact_type=StorageArtifactType.ARTIFACT,
            )
            artifact_response = self.storage_manager.download(
                ers_key=artifact_ers_key, storage_dict=self.storage
            )

            if response:
                model_local_storage_savepath = self.storage_manager.getLocalSavePath(
                    ers_key=model_ers_key
                )
                # TODO replace with ModelAbstract's own loader?????
                if self.gpus == 0:
                    self.model.load_state_dict(
                        torch.load(
                            model_local_storage_savepath, map_location=self.device
                        )
                    )
                else:
                    self.model.load_state_dict(torch.load(model_local_storage_savepath))
                self.logger.info(
                    "Finished loading model state_dict from %s"
                    % model_local_storage_savepath
                )
                if artifact_response:
                    # we have the file

                    artifact_local_storage_savepath = (
                        self.storage_manager.getLocalSavePath(ers_key=artifact_ers_key)
                    )

                    checkpoint = torch.load(artifact_local_storage_savepath)
                    for pgn in self.parameter_groups:
                        self.optimizer[pgn].load_state_dict(
                            checkpoint["optimizer"][pgn]
                        )

                        self.scheduler[pgn].load_state_dict(
                            checkpoint["scheduler"][pgn]
                        )
                    self.logger.info(
                        "Finished loading optimizer state_dict from %s"
                        % artifact_local_storage_savepath
                    )
                    self.logger.info(
                        "Finished loading scheduler state_dict from %s"
                        % artifact_local_storage_savepath
                    )

                    for lossname in self.loss_fn:
                        self.loss_fn[lossname].load_state_dict(
                            checkpoint["loss_fn"][lossname]
                        )
                        if self.loss_optimizer[lossname] is not None:
                            self.loss_optimizer[lossname].load_state_dict(
                                checkpoint["loss_optimizer"][lossname]
                            )
                        if self.loss_scheduler[lossname] is not None:
                            self.loss_scheduler[lossname].load_state_dict(
                                checkpoint["loss_scheduler"][lossname]
                            )

                        self.logger.info(
                            "Finished loading loss state_dict from %s"
                            % artifact_local_storage_savepath
                        )
                else:
                    # we have model but not artifact. We will load model.
                    if not ignore_if_error:
                        raise FileNotFoundError(
                            "Could not download artifact %s from epoch %i ? step %i"
                            % (
                                StorageArtifactType.ARTIFACT.value,
                                load_epoch,
                                load_step,
                            )
                        )
            else:
                if not ignore_if_error:
                    raise FileNotFoundError(
                        "Could not download artifact %s from epoch %i ? step %i"
                        % (StorageArtifactType.MODEL.value, load_epoch, load_step)
                    )
                return False

        elif artifact == StorageArtifactType.ARTIFACT:
            raise NotImplementedError("MODEL_ARTIFACT is loaded with MODEL")
        elif artifact == StorageArtifactType.CONFIG:
            raise NotImplementedError(
                "CONFIG downloads should not be managed inside Trainer"
            )
        elif artifact == StorageArtifactType.LOG:
            raise NotImplementedError(
                "LOG downloads should not be managed inside Trainer"
            )
        elif artifact == StorageArtifactType.PLUGIN:
            raise NotImplementedError("PLUGIN downloads should be used with Deploy")
        elif artifact == StorageArtifactType.METRIC:
            raise NotImplementedError(
                "METRIC downloads should not be managed inside Trainer"
            )
        else:
            raise ValueError("Unexpected value for artifact type %s" % artifact.value)

        return True

    def loadModel(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadArtifact(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadPlugin(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadCode(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadConfig(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadLog(epoch: int, step: int, ignore_if_error: bool):
        pass

    def loadMetrics(epoch: int, step: int, ignore_if_error: bool):
        pass

    def train(self, continue_epoch: int = None, continue_step: int = None, **kwargs):
        ers_key = self.storage_manager.getLatestERSKey(
            artifact=StorageArtifactType.MODEL
        )
        if continue_epoch is None:  # Use the provided latest key
            self.logger.debug(
                "`continue_epoch` is not provided. Will use latest `ers_key`"
            )
            continue_epoch = ers_key.storage.epoch
            continue_step = ers_key.storage.step
            self.current_ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        else:  # contnue epoch and/or step are provided
            # Check if they are valid. Otherwise, default to provided latest_key
            self.logger.debug(
                "`continue_epoch` is provided. Checking validity in remote and local Storage with artifact MODEL"
            )
            key_exist = self.storage_manager.checkEpoch(
                storage=self.storage,
                epoch=continue_epoch,
                artifact=StorageArtifactType.MODEL,
            )
            if key_exist is False:
                self.logger.debug(
                    "`continue_epoch` value of %i does not exist. Will switch to latest ERSKey."
                    % continue_epoch
                )
                continue_epoch = None
            else:  # Epoch exists, we now need to check if the step value is valid
                self.logger.debug(
                    "`continue_epoch` value of %i is valid. Checking `continue_step`."
                    % continue_epoch
                )
                if continue_step is not None:
                    key_exist = self.storage_manager.checkStep(
                        storage=self.storage,
                        epoch=continue_epoch,
                        step=continue_step,
                        artifact=StorageArtifactType.MODEL,
                    )
                    if key_exist is False:
                        self.logger.debug(
                            "`continue_step` value of %i does not exist. Will attempt to find latest step."
                            % continue_step
                        )
                        continue_step = None
                    else:
                        self.logger.debug(
                            "`continue_step` value of %i is valid." % continue_step
                        )
                # continue_step is None if not provided, or if the provided value is not valid.
                if continue_step is None:
                    self.logger.debug(
                        "`continue_step` is not provided or not valid. Getting latest step saved in Epoch %i in remote and local Storage"
                        % continue_epoch
                    )
                    key = self.storage_manager.getLatestStepOfArtifactWithEpoch(
                        storage=self.storage,
                        epoch=continue_epoch,
                        artifact=StorageArtifactType.MODEL,
                    )
                    if key is None:
                        self.logger.debug(
                            "No latest step found even though `continue_epoch` is valid. Will switch to latest ERSKey."
                        )
                        continue_step = None
                        continue_epoch = None
                    else:
                        self.logger.debug(
                            "Found latest `continue_step` %i" % key.storage.step
                        )
                        continue_step = key.storage.step

            if (continue_epoch is not None) and (continue_step is not None):
                self.logger.info(
                    "Using provided epoch/step pair %i/%i."
                    % (continue_epoch, continue_step)
                )
                self.current_ers_key = self.storage_manager.getERSKey(
                    epoch=continue_epoch, step=continue_step
                )
            else:
                continue_epoch = ers_key.storage.epoch
                continue_step = ers_key.storage.step
                self.current_ers_key = KeyMethods.cloneERSKey(ers_key)
            self.logger.info(
                "Using ERSKey {key}".format(key=self.current_ers_key.printKey())
            )

        # This occurs ONLY if using latestStorageKey and it is empty. TODO this should be
        # cleaned up for potential bugs when continue_epoch is -1 for other reasons due to
        # leaks but storageManager is still forced to update StorageKey
        if continue_epoch == -1:
            self.logger.info("Starting from scratch. ")
            self.logger.info(
                "Switching to latest ERSKey `{key}`".format(
                    key=self.storage_manager.getLatestERSKey().printKey()
                )
            )
            self.storage_manager.updateStorageKey(self.storage_manager.getNextERSKey())
            self.logger.info(
                "Incremented latest ERSKey to `{key}`".format(
                    key=self.storage_manager.getLatestERSKey().printKey()
                )
            )
            self.current_ers_key = self.storage_manager.getLatestERSKey()
            continue_epoch = self.current_ers_key.storage.epoch
            continue_step = self.current_ers_key.storage.step
        else:
            self.logger.info(
                "Starting training. with `continue_epoch` %i and `continue_step` %i"
                % (continue_epoch, continue_step)
            )
            if self.edna_context.MODEL_HAS_LOADED_WEIGHTS:
                self.logger.info(
                    "Weights have already been loaded into model. Skipping loading of epoch-specific weights from Epoch %i Step %i"
                    % (continue_epoch, continue_step)
                )
            else:
                self.logger.info(
                    "Attempting to load weights from from Epoch %i Step %i"
                    % (continue_epoch, continue_step)
                )
                # Attempt to load models, with skip-if-error
                response = self.load(
                    load_epoch=continue_epoch,
                    load_step=continue_step,
                    artifact=StorageArtifactType.MODEL,
                    ignore_if_error=True,
                )
                if not response:
                    # This is for non-initial epoch/step
                    if continue_epoch or continue_step:
                        raise RuntimeError(
                            "Could not load weights at epoch-step %i/%i."
                            % (continue_epoch, continue_step)
                        )
                    # self.logger.info()
                    self.logger.info(
                        "Could not load weights since there is no model saved yet."
                    )
                else:  # If we have loaded weights from a specific epoch, we should continue from the next epoch...
                    self.logger.info(
                        "Loaded weights at epoch-step %i/%i."
                        % (continue_epoch, continue_step)
                    )

            self.logger.info(
                "Switching to latest ERSKey `{key}`".format(
                    key=self.storage_manager.getLatestERSKey().printKey()
                )
            )
            self.storage_manager.updateStorageKey(self.storage_manager.getNextERSKey())
            self.logger.info(
                "Incremented latest ERSKey to `{key}`".format(
                    key=self.storage_manager.getLatestERSKey().printKey()
                )
            )
            self.current_ers_key = self.storage_manager.getLatestERSKey()
            continue_epoch = self.current_ers_key.storage.epoch
            continue_step = self.current_ers_key.storage.step
        if continue_step == -1:
            raise RuntimeError("`continue_step` is -1 after error checking")

        self.logger.info("Logging to:\t%s" % self.log_manager.getLocalLog())
        self.logger.info(
            "Models will be saved locally with base name:\t%s"
            % self.storage_manager.getLocalSavePath(
                self.storage_manager.getERSKey(
                    epoch=0, step=0, artifact_type=StorageArtifactType.MODEL
                )
            )
        )
        self.logger.info(
            "Model artifacts will be saved locally with base name:\t%s"
            % self.storage_manager.getLocalSavePath(
                self.storage_manager.getERSKey(
                    epoch=0, step=0, artifact_type=StorageArtifactType.ARTIFACT
                )
            )
        )
        self.logger.info(
            "Logs will be saved locally with base name:\t%s"
            % self.storage_manager.getLocalSavePath(
                self.storage_manager.getERSKey(
                    epoch=0, step=0, artifact_type=StorageArtifactType.LOG
                )
            )
        )
        self.logger.info(
            "Metrics will be saved locally with base name:\t%s"
            % self.storage_manager.getLocalSavePath(
                self.storage_manager.getERSKey(
                    epoch=0, step=0, artifact_type=StorageArtifactType.METRIC
                )
            )
        )
        self.logger.info("Config will not be saved")
        self.logger.info("Plugins will not be saved")

        if not self.skipeval:
            self.logger.info("Performing initial evaluation...")
            self.initial_evaluate()
        else:
            self.logger.info("Skipping initial evaluation.")
        self.logger.info("Training for %i epochs" % self.epochs)
        self.logger.info("Starting training from Epoch %i" % continue_epoch)
        self.beginning_of_training_hook()
        self.zeroGrad()
        self.evaluateFlag = False
        self.saveFlag = False
        for epoch in range(self.epochs):
            if epoch >= continue_epoch:
                # TODO pre epoch
                self.epoch_step(epoch)
                # TODO post
                self.global_epoch += 1
            else:
                self.global_epoch = epoch + 1

        if self.evaluateFlag:
            self.logger.info("Final: Evaluating model at test-frequency")
            self.evaluate(save_log=True)
            self.evaluateFlag = False
        if self.saveFlag:
            self.logger.info(
                "Final: Saving model at save-frequency. Saving Logs as well."
            )
            self.save(self.saveFlag_epoch, self.saveFlag_step, local_only = self.saveFlag_localonly)
            self.save(
                self.saveFlag_epoch,
                self.saveFlag_step,
                artifact=StorageArtifactType.LOG,
                local_only = self.saveFlag_localonly
            )
            self.saveFlag = False
        self.logger.info("Finished training")

    def initial_evaluate(self):
        """Evaluation of model before we start training"""
        self.evaluate()

    def epoch_step(self, epoch):
        """Trains model for an epoch."""
        self.global_batch = 0
        for batch in self.train_loader:
            if self.global_batch == 0:
                self.printOptimizerLearningRates()

            self.model.train()  # train == we are tracking all numbers and computation graph
            batch = self.move_to_device(batch)
            loss: torch.Tensor = self.step(batch)  # perform function and returns loss
            self.model_params["raw_loss"] = loss.cpu().item()
            loss = loss / self.accumulation_steps
            self.model_params["scaled_loss"] = loss.cpu().item()
            self.model_params["loss"] = self.model_params["scaled_loss"]
            loss.backward()
            self.accumulation_count += 1
            # TODO params pattern as property!!!!
            self.model_params["accumulation_count"] = self.accumulation_count

            if self.accumulation_count % self.accumulation_steps == 0:
                self.updateGradients()
                self.accumulation_count = 0
                if self.evaluateFlag:
                    self.logger.info("Evaluating model at test-frequency")
                    self.evaluate(save_log=True)
                    self.evaluateFlag = False
                if self.saveFlag:
                    self.logger.info(
                        "Saving model at save-frequency, at epoch %i, step %i"
                        % (self.saveFlag_epoch, self.saveFlag_step)
                    )
                    self.save(self.saveFlag_epoch, self.saveFlag_step, local_only = self.saveFlag_localonly)
                    self.saveFlag = False

            if self.storage_manager.storage_trigger_strict:
                self.check_step_save(self.global_batch)
            else:  # We check every step_verbose steps
                if (self.global_batch + 1) % self.step_verbose == 0:
                    self.check_step_save(self.global_batch + 1)
            self.end_of_step_metrics()
            self.updateStepMetrics(self.global_epoch, self.global_batch)
            if (self.global_batch + 1)%self.step_verbose == 0:
                self.updateBatchMetrics(self.global_epoch, self.global_batch)


            if (self.global_batch + 1) % self.step_verbose == 0:
                self.printStepInformation()
            self.global_batch += 1
            # if self.step_save_frequency and self.global_batch % self.step_save_frequency == 0:
            #     self.set_save_flag()

        self.stepSchedulers()
        self.stepLossSchedulers()

        if self.global_epoch % self.test_frequency == 0:
            if self.accumulation_steps > 0:
                self.logger.info(
                    "Model evaluation triggered, but gradients still need"
                    " accumulation. Will evaluate after accumulation."
                )
                self.evaluateFlag = True
            else:
                self.evaluate(save_log=True)

        self.check_epoch_save(self.global_epoch)
        # if self.global_epoch % self.save_frequency == 0:
        #     if self.accumulation_steps > 0:
        #         self.logger.info(
        #             "Model save triggered, but gradients still need"
        #             " accumulation. Will save after accumulation."
        #         )
        #         self.set_save_flag()
        #     else:
        #         self.save()
        #self.updateStepMetrics(self.global_epoch, self.global_batch)   # ALREADY DONE EARLIER
        self.updateBatchMetrics(self.global_epoch, self.global_batch)
        self.logger.info(
            "{0} Completed epoch {1} {2}".format("*" * 10, self.global_epoch, "*" * 10)
        )
    

    def end_of_step_metrics(self):
        """Log any additional metrics at the end of a step
        """
        pass

    def beginning_of_training_hook(self):
        pass



    def log_metric(self, metric_name, metric_val):
        self.metrics_manager.log_metric(
            metric_name=metric_name,epoch=self.global_epoch, step=self.global_batch, value=metric_val,metric_type="model"
        )

    # Check whether to save each of the artifacts at the current global_batch step.
    # If so, perform a save (or in case of model, plugin, and artifact, perform a setSaveFlag() for the accumulation_step bit)
    # TODO need to check whether accumulation_step < step_frequency for model stuff, because that will cause issues
    # Additional notes: for model and artifact saving, we need a way to tie their frequency together
    # That is, we will completely ignore whatever frequency is for ARTIFACT, because we save artifacts WHEN
    # we save model; we just save it in the artifact location, rather than the model location
    def check_step_save(self, step):
        # MODEL and MODEL_ARTIFACTS
        local_save = self.storage_manager.getSaveTriggerForStep(step=step)
        model_upload = self.storage_manager.getUploadTriggerForStep(step, StorageArtifactType.MODEL)
        if local_save or model_upload:
            # For gradient accumulation
            if model_upload:
                self.set_save_flag(epoch=self.global_epoch, step=step, local_only = False)
            else:
                self.set_save_flag(epoch=self.global_epoch, step=step, local_only = True)

        if local_save:
            import pdb
            pdb.set_trace()
            self.save(artifact=StorageArtifactType.PLUGIN, save_step=step, local_only=self.storage_manager.getUploadTriggerForStep(step, StorageArtifactType.PLUGIN))
            self.save(artifact=StorageArtifactType.LOG, save_step=step, local_only=self.storage_manager.getUploadTriggerForStep(step, StorageArtifactType.LOG))
            self.save(artifact=StorageArtifactType.METRIC, save_step=step, local_only=self.storage_manager.getUploadTriggerForStep(step, StorageArtifactType.METRIC))
            self.save(artifact=StorageArtifactType.CONFIG, save_step=step, local_only=self.storage_manager.getUploadTriggerForStep(step, StorageArtifactType.CONFIG))


    def check_epoch_save(self, epoch):  # TODO off by one errors
        self.logger.debug("Checking epoch save status at Epoch %i" % epoch)
        
        # MODEL and MODEL_ARTIFACTS
        local_save = self.storage_manager.getSaveTriggerForEpoch(epoch=epoch)
        model_upload = self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.MODEL)
        if local_save or model_upload:
            # For gradient accumulation
            if model_upload:
                self.set_save_flag(epoch=epoch, step=self.global_batch, local_only = False)
            else:
                self.set_save_flag(epoch=epoch, step=self.global_batch, local_only = True)
        

        if local_save:
            self.save(artifact=StorageArtifactType.PLUGIN, save_epoch=epoch, local_only=self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.PLUGIN))
            self.save(artifact=StorageArtifactType.LOG, save_epoch=epoch, local_only=self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.LOG))
            self.save(artifact=StorageArtifactType.METRIC, save_epoch=epoch, local_only=self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.METRIC))
            self.save(artifact=StorageArtifactType.CONFIG, save_epoch=epoch, local_only=self.storage_manager.getUploadTriggerForEpoch(epoch, StorageArtifactType.CONFIG))


    def set_save_flag(self, epoch, step, local_only = False):
        self.saveFlag = True
        self.saveFlag_epoch = epoch
        self.saveFlag_step = step
        self.saveFlag_localonly = local_only

    def set_evaluate_flag(self):
        pass

    def move_to_device(self, batch) -> Tuple[torch.Tensor]:
        return (item.to(self.device) for item in batch)

    def step(self, batch) -> torch.Tensor:
        # compute the loss, and return it
        # print("!!!!!!!!!! batch !!!!!!!!!!!!!!!",batch)

        raise NotImplementedError()

    def updateGradients(self):
        self.stepOptimizers()
        self.stepLossOptimizers()
        self.zeroGrad()

    def zeroGrad(self):
        self.zeroGradOptimizers()
        self.zeroGradLossOptimizers()

    def printStepInformation(self):
        
        
        print_metrics = "\t".join(["{mname}:  {mval}".format(mname=metric_name, mval=self.metrics_manager.metrics[metric_name].getLastVal()) for metric_name in self.config.LOGGING.PRINT_METRICS])


        self.logger.info(
            "Epoch{0}.{1} Completed. Metrics: {metrics}".format(
                self.global_epoch, self.global_batch, metrics = print_metrics
            ))

    def evaluate(self, save_log = False):
        self.model.eval()
        logit_labels, true_labels, features = self.evaluate_impl()
        if save_log:
            self.saveLog()
        return logit_labels, true_labels, self.crawler.classes, features

    def evaluate_impl(self):
        raise NotImplementedError()

    def zeroGradOptimizers(self):
        for optim in self.optimizer:
            self.optimizer[optim].zero_grad()

    def zeroGradLossOptimizers(self):
        for lossname in self.loss_fn:
            if self.loss_optimizer[lossname] is not None:
                self.loss_optimizer[lossname].zero_grad()

    def stepOptimizers(self):
        for optim in self.optimizer:
            self.optimizer[optim].step()

    def stepLossOptimizers(self):
        for lossname in self.loss_fn:
            if (
                self.loss_optimizer[lossname] is not None
            ):  # In case loss object doesn;t have any parameters, this will be None. See optimizers.StandardLossOptimizer
                self.loss_optimizer[lossname].step()

    def stepSchedulers(self):
        for scheduler in self.scheduler:
            self.scheduler[scheduler].step()

    def stepLossSchedulers(self):
        for lossname in self.loss_fn:
            if self.loss_scheduler[lossname] is not None:
                self.loss_scheduler[lossname].step()

    def printOptimizerLearningRates(self):
        for param_group_name in self.optimizer:
            lrs = self.scheduler[param_group_name].get_last_lr()
            lrs = sum(lrs) / float(len(lrs))
            self.logger.info(
                "Parameter Group `{0}`: Starting epoch {1} with {2} steps and"
                " learning rate {3:2.5E}".format(
                    param_group_name,
                    self.global_epoch,
                    len(self.train_loader) - (len(self.train_loader) % 10),
                    lrs,
                )
            )
