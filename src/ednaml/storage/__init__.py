import logging
from typing import Dict, Union
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from ednaml.config.SaveConfig import SaveConfig
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.storage.AzureStorage import AzureStorage
from ednaml.storage.LocalStorage import LocalStorage
from ednaml.utils.SaveMetadata import SaveMetadata
from ednaml.utils import StorageArtifactType, ExperimentKey, RunKey, StorageKey, ERSKey, StorageNameStruct
import os

class StorageManager:  
    """StorageManager is a helper class for storage-related tasks in EdnaML

    It has 4 tasks:
    1. Provide a ERSKey given the epoch, run, step, and type parameters
    2. Provide a local-file-name given a StorageNameStruct
    3. Provide a trigger-mechanism for checking upload requirements given the configuration file and current epoch and step
    4. Provide the storage name for a given upload type (e.g. config, logfile, model, etc)
    """
    def __init__(self,  logger: logging.Logger, 
                        cfg: EdnaMLConfig, 
                        experiment_key: ExperimentKey, 
                        storage_manager_mode = "loose"):
        self.logger = logger
        
        self.experiment_key = experiment_key
        self.storage_manager_mode = storage_manager_mode    # strict or loose
        self.log("Initializing StorageManager")
        self.log("\tusing experiment_key: \t{ekey}".format(ekey=self.experiment_key.getExperimentName()))
        self.log("\twith storage_manager_mode: \t{mode}".format(mode=self.storage_manager_mode))
        self.cfg = cfg

        self.run_key: RunKey = None
        self.latest_storage_key: StorageKey = None
        # Add options here for where to save local files, i.e. not directly in ./
        self.local_save_directory = "%s-v%s-%s-%s" % self.experiment_key.getKey()
        self.log("\tUsing local save directory: \t%s"%self.local_save_directory)
        
        # So, the log file is still not set up
        # We do it when the run is initialized.
        

        # Runs are handled elsewhere, so without it, we have essentially a misconfigured directory
        # TODO potentially problematic...?
        # But we cannot do this unless storage is packaged with StorageManager
        # So for now, in EdnaML, we do: buildStorageManager -> buildStorage -> handleRuns
        
        # I.e., when StorageManager is initialized, we check whether backup is set to true
        # Then we check each backup to see if they already have a run
        # If so, we get the max run, and increment by one
        # ONLY if specified. Default behavior is to continue from prior run
        # BUT, we can choose to execute a new run...
        os.makedirs(self.local_save_directory, exist_ok=True)
        # self.file_basename = "_".join([self.metadata.MODEL_CORE_NAME, 
        #                     "v%i"%self.metadata.MODEL_VERSION,
        #                     self.metadata.MODEL_BACKBONE,
        #                     self.metadata.MODEL_QUALIFIER])
        self.file_basename = "experiment"
        self.log("\tUsing file basename: \t%s"%self.file_basename)
        self.local_storage = LocalStorage(experiment_key=self.experiment_key,
            storage_name = "ednaml-local-storage-reserved", storage_url="./", file_basename = self.file_basename)
        self.log("Generated `ednaml-local-storage-reserved` LocalStorage object")
        # We create a fast, O(1) reference for each of the save-options to avoid switch statements later
        self.artifact_references = {
            StorageArtifactType.MODEL: self.cfg.SAVE.MODEL_BACKUP,
            StorageArtifactType.ARTIFACT: self.cfg.SAVE.ARTIFACTS_BACKUP,
            StorageArtifactType.PLUGIN: self.cfg.SAVE.PLUGIN_BACKUP,
            StorageArtifactType.METRIC: self.cfg.SAVE.METRICS_BACKUP,
            StorageArtifactType.CONFIG: self.cfg.SAVE.CONFIG_BACKUP,
            StorageArtifactType.LOG: self.cfg.SAVE.LOG_BACKUP,
        }

        class LooseTriggerMethod:
            def __init__(self, trigger_frequency: int, initial_state: int = -1, base = 0):
                self.trigger_frequency: int = trigger_frequency
                self.state = initial_state
                self.base = base
            def __call__(self, check_value: int)-> bool:
                # Get the quotient, i.e. whether we are at a multiplicative factor of the trigger_frequency
                cv = int(check_value/self.trigger_frequency)
                if cv != self.state:    # Check if same multiplicative factor as before
                    self.state = cv
                    if cv>=self.base:   # Check if we are above the lowest threshold. 0 for EPOCH, 1 for STEP.
                        return True
                return False

        # Trigger methods tell us whether to trigger an upload or not. To improve speed, we cache many of the parameters here.
        if self.storage_manager_mode == "strict":
        # We create a fast O(1) reference to the trigger checking methods here for epochs
            self.epoch_triggers = {
                StorageArtifactType.MODEL: (lambda x: False) if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.ARTIFACT: (lambda x: False) if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.PLUGIN: (lambda x: False) if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.METRIC: (lambda x: False) if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.CONFIG: (lambda x: False) if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.LOG: (lambda x: False) if self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0 else (lambda x: (x%self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0)),
            }

        elif self.storage_manager_mode == "loose":
            self.epoch_triggers = {
                StorageArtifactType.MODEL: (lambda x: False) if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.MODEL_BACKUP.FREQUENCY),
                StorageArtifactType.ARTIFACT: (lambda x: False) if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY),
                StorageArtifactType.PLUGIN: (lambda x: False) if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY),
                StorageArtifactType.METRIC: (lambda x: False) if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.METRICS_BACKUP.FREQUENCY),
                StorageArtifactType.CONFIG: (lambda x: False) if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY),
                StorageArtifactType.LOG: (lambda x: False) if self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.LOG_BACKUP.FREQUENCY),
            }
        else:
            raise NotImplementedError()

        self.log("Generated EpochTrigger checks")
        if self.storage_manager_mode == "strict":
        # We create a fast O(1) reference to the trigger checking methods here for epochs
            self.step_triggers = {
                StorageArtifactType.MODEL: (lambda x: False) if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.ARTIFACT: (lambda x: False) if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.PLUGIN: (lambda x: False) if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.METRIC: (lambda x: False) if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.CONFIG: (lambda x: False) if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.LOG: (lambda x: False) if self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0 else (lambda x: (x%self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0)),
            }

        elif self.storage_manager_mode == "loose":
            self.step_triggers = {
                StorageArtifactType.MODEL: (lambda x: False) if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP, base=1),
                StorageArtifactType.ARTIFACT: (lambda x: False) if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP, base=1),
                StorageArtifactType.PLUGIN: (lambda x: False) if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP, base=1),
                StorageArtifactType.METRIC: (lambda x: False) if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP, base=1),
                StorageArtifactType.CONFIG: (lambda x: False) if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP, base=1),
                StorageArtifactType.LOG: (lambda x: False) if self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0 else LooseTriggerMethod(self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP, base=1),
            }
        else:
            raise NotImplementedError()
        self.log("Generated StepTrigger checks")

    def log(self, msg):
        self.logger.debug("[StorageManager] %s"%msg)

    def getERSKey(self, epoch: int, step: int, artifact_type: StorageArtifactType = StorageArtifactType.MODEL) -> ERSKey:
        """Given epoch, run, step, and artifact type, we will construct a StorageKey
        (a StorageNameStruct) and return it.

        Args:
            epoch (_type_): _description_
            run (_type_): _description_
            step (_type_): _description_
            artifact_type (StorageArtifactType): _description_
        """
        return ERSKey(
            self.experiment_key, self.run_key, StorageKey(epoch, step, artifact_type)
        )

    def getExperimentKey(self) -> ExperimentKey:
        return self.experiment_key

    def getRunKey(self) -> RunKey:
        return self.run_key

    def getLatestStorageKey(self) -> StorageKey:
        return self.latest_storage_key

    def getLatestERSKey(self, artifact: StorageArtifactType = StorageArtifactType.MODEL) -> ERSKey:
        return self.getERSKey(epoch = self.latest_storage_key.epoch, step = self.latest_storage_key.step, artifact_type=artifact)

    def getNextERSKey(self, artifact: StorageArtifactType = StorageArtifactType.MODEL) -> ERSKey:
        return self.getERSKey(epoch = self.latest_storage_key.epoch + 1, step = 0, artifact_type=artifact)

    def download(self, storage_dict: Dict[str, BaseStorage], ers_key: ERSKey) -> bool:
        """Download the file(s) corresponding to the ERSKey if they do not already exist locally

        Args:
            storage_dict (Dict[str, BaseStorage]): _description_
            ers_key (ERSKey): _description_
        """
        local_path = self.getLocalSavePath(ers_key=ers_key)
        if not os.path.exists(local_path):
            return storage_dict[self.getStorageNameForArtifact(ers_key.storage.artifact)].download(ers_key=ers_key, 
                destination_file_name=local_path)
        return True # Already exists.

    def upload(self, storage_dict: Dict[str, BaseStorage], ers_key: ERSKey) -> bool:
        """Upload the file(s) corresponding to the ERSKey. If they already exist, Storage will throw an error.

        Args:
            storage_dict (Dict[str, BaseStorage]): _description_
            ers_key (ERSKey): _description_
        """
        source_file_name = self.getLocalSavePath(ers_key=ers_key)
        if os.path.exists(source_file_name):
            storage_dict[self.getStorageNameForArtifact(ers_key.storage.artifact)].upload(ers_key=ers_key, 
                source_file_name=source_file_name)
            return True
        return False
        


    def getLocalFileName(self, ers_key: ERSKey) -> Union[str,os.PathLike]:
        """Creates the local file name for the provided StorageKey. This does not contain experiment details, or run details.

        Args:
            storage_struct (StorageNameStruct): The StorageKey to construct a local file name for

        Returns:
            Union[str,os.PathLike]: The constructed file name that combines attributes of the StorageKey
        """
        return self.local_storage.path_of_artifact(ers_key.storage.epoch, ers_key.storage.step, ers_key.storage.artifact)
    
    def getLocalSavePath(self, ers_key: ERSKey) -> Union[str, os.PathLike]:
        """Provides the complete path to the file name for the provided StorageKey

        Args:
            ers_key (ERSKey): _description_

        Returns:
            Union[str, os.PathLike]: _description_
        """
        return os.path.join(self.local_storage.storage_path, self.local_storage.run_dir, self.getLocalFileName(ers_key=ers_key))

    def getUploadTriggerForEpoch(self, epoch: int, artifact_type: StorageArtifactType) -> bool:
        """Determines whether an upload should occur at this epoch given the 
        provided StorageArtifactType.

        Args:
            epoch (int): Epoch number
            artifact_type (StorageArtifactType): The artifact type for this trigger check.

        Returns:
            bool: If true, we should upload. If false, we should not upload.
        """
        return self.epoch_triggers[artifact_type](epoch)

    def getUploadTriggerForStep(self, step: int, artifact_type: StorageArtifactType) -> bool:
        return self.step_triggers[artifact_type](step)
        
    def performBackup(self, artifact_type: StorageArtifactType) -> bool:
        return self.artifact_references[artifact_type].BACKUP

    def getStorageNameForArtifact(self, artifact_type: StorageArtifactType) -> str:
        return self.artifact_references[artifact_type].STORAGE_NAME


    def setTrackingRun(self, storage_dict: Dict[str, BaseStorage] = None, tracking_run: int = None, new_run: bool = False):
        self.log("Tracking run with `new_run`: %s"%str(new_run))
        if tracking_run is None:
            max_run_list = [storage_dict[self.getStorageNameForArtifact(artifact_key)].getMaximumRun() if self.performBackup(artifact_key) else -1 for artifact_key in self.artifact_references]
            max_run = max(max_run_list)
            if max_run == -1:
                tracking_run = 0
            else:
                tracking_run = max_run + int(new_run)
            self.log("Remote tracking run is %i"%tracking_run)

            local_tracking_run = self.local_storage.getMaximumRun() + int(new_run)
            self.log("Local tracking run is %i"%tracking_run)
            tracking_run = max(tracking_run, local_tracking_run)

        self.log("Tracking run in `ednaml-local-storage-reserved` set to: %i"%tracking_run)
        
        # NOTE at this time, we ignore all this complication, and just save the config in the run directly.
        # Storage's uploadConfig handles doubles by renaming the existing config by including the most recent StorageKey from saved model(s)
        # Then, the provided config is uploaded
        self._setTrackingRun(storage_dict, tracking_run)

        #ers_key = self.getERSKey(epoch = 0, step = 0, artifact_type=StorageArtifactType.CONFIG)
        #self.cfg.save(self.getLocalSavePath(ers_key=ers_key))
        #storage_dict[self.getStorageNameForArtifact(StorageArtifactType.CONFIG)].uploadConfig(ers_key = ers_key, 
        #                                                                                        source_file_name = self.getLocalSavePath(ers_key=ers_key)) 
        
        # TODO Code upload by zipping files. Skip for now...
        
        
        
        # TODO
        # Now we need to handle the case where the existing config in the run and the current config do not match
        """
        if not new_run:
            # First check config difference.
            # Strict: Use current max tracking_run iff remote and local configurations match exactly
            # Loose: Use current max tracking_run regardless of different between local and remote configurations. When saving, replace config
            # Flexible: Use current max tracking_run iff only in case of specific key differences (e.g. batch size, workers). For other differences, 
            #           generate a new run, BUT we want to keep a back-reference in the saved config to the chained ExperimentKey
            # That is, config should have a new section, for tracking when we start off from another experiment itself
            # That way, if we are requesting a specific epoch-step, and it does not exist, we can look back in the chain to see if it exists
            # in the parent ExperimentKey. But, this is super duper complicated.
            pass
            
            ers_key = self.getERSKey(epoch = 0, step = 0, artifact_type=StorageArtifactType.CONFIG)
            storage_dict[self.getStorageNameForArtifact(StorageArtifactType.CONFIG)].downloadConfig(ers_key = ers_key, local_file_name = os.path.join(self.local_save_directory, "configuration_remote.yml")) 
            self.cfg.save(os.path.join(self.local_save_directory, "configuration_current.yml"))
            require_new_run = EdnaMLConfig.compare(
                os.path.join(self.local_save_directory, "configuration_current.yml"),
                os.path.join(self.local_save_directory, "configuration_remote.yml"),
                mode = config_mode 
            )
            if require_new_run
        """

        

    def _setTrackingRun(self, storage_dict: Dict[str, BaseStorage], tracking_run: int):
        self.run_key = RunKey(run=tracking_run)

        # Create the run directory locally.
        self.local_storage.setTrackingRun(tracking_run)


        # Inform the storages that we have a run:
        for artifact_key in self.artifact_references:
            if self.performBackup(artifact_type=artifact_key):
                storage_name = self.getStorageNameForArtifact(artifact_key)
                storage_dict[storage_name].setTrackingRun(tracking_run)
                self.log("Tracking run for Artifact `%s` at Storage `%s` set to: %i"%(artifact_key.value, storage_name, tracking_run))

        # Copy the files over??? Or save that for other methods to perform...


    def setLatestStorageKey(self, storage_dict: Dict[str, BaseStorage], artifact: StorageArtifactType = None):
        """Check local storage as well as remote storage(s) for the latest epoch-step pair when something was saved.

        Args:
            storage_dict (Dict[str, BaseStorage]): _description_
            artifact (StorageArtifactType, optional): _description_. Defaults to None.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """
        # Default to checking with MODEL, TODO add functionality in Storage to handle other artifact types.
        if artifact is None:
            artifact = StorageArtifactType.MODEL
        
        self.log("Intializing reference StorageKey to (-1,-1), with reference artifact: %s"%artifact.value)
        ers_key: ERSKey = self.getERSKey(epoch = -1, step = -1, artifact_type=artifact)
        remote_ers_key = storage_dict[self.getStorageNameForArtifact(artifact_type=artifact)].getLatestStorageKey(ers_key)
        local_ers_key = self.local_storage.getLatestStorageKey(ers_key=ers_key)

        self.log("Found remote ERSKey with reference artifact: %s \n %s"%(artifact.value, repr(remote_ers_key)))
        self.log("Found local ERSKey with reference artifact: %s \n %s"%(artifact.value, repr(local_ers_key)))
        


        final_ers_key = None
        if local_ers_key.storage.epoch > remote_ers_key.storage.epoch:
            final_ers_key = local_ers_key
        elif remote_ers_key.storage.epoch > local_ers_key.storage.epoch:
            final_ers_key = remote_ers_key
        elif local_ers_key.storage.epoch == remote_ers_key.storage.epoch:
            if local_ers_key.storage.step > remote_ers_key.storage.step:
                final_ers_key = local_ers_key
            elif remote_ers_key.storage.step > local_ers_key.storage.step:
                final_ers_key = remote_ers_key
            elif local_ers_key.storage.step == remote_ers_key.storage.step:
                final_ers_key = local_ers_key
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()

        self.log("Obtained latest StorageKey at (%i,%i), with reference artifact: %s"%(final_ers_key.storage.epoch, final_ers_key.storage.step, artifact.value))
        self.latest_storage_key = StorageKey(epoch = final_ers_key.storage.epoch,
                                            step = final_ers_key.storage.step,
                                            artifact=artifact)


    def updateStorageKey(self, ers_key: ERSKey):
        self.latest_storage_key.epoch = ers_key.storage.epoch
        self.latest_storage_key.step = ers_key.storage.step

        

    def initializeLog(self, storage_dict: Dict[str, BaseStorage]):
        """Initialize a log file (or append to an existing one / log-server)

        Ok, so we have some logging solution -- how does this look like?

        presumably, StorageManager contains a LocalLogManager (instead of python logging)
            LocalLogManager --> physical logs, e.g. logging.Log
            LocalLogManager --> Logstash logs through the logstash logging handler
            So, basically, we can set up what the logging solution is in config directly

            In initialize log, we do the actual logstash/physical log setup (because logstash will need
            some key to reference this log, right, for our experiment-key + run-key)

            NOTE: say we are using LogStash -- then we likely do not need a LogBackup
            Alternatively, one can use LogStash + LogBackup
            Or use the regular file storage + LogBackup Storage instantiates a background process
            to manage the logging continuously without needing manual triggering...



        Returns:
            _type_: _description_
        """
        if self.performBackup(StorageArtifactType.LOG):
            # We are performing log backups.
            # We will use Storage to download the current log state
            # Note: not every log will do this ! Remote loggers can just ignore...
            storage_dict[self.getStorageNameForArtifact(StorageArtifactType.LOG)].download(
                file_struct = self.getERSKey(epoch=0,step=0,artifact_type=StorageArtifactType.LOG), 
                destination_file_name = self.path_ends[StorageArtifactType.LOG]((0, 0)) # epoch and step do not matter for logs...
            )
        
        # No backup being performed.
        # Directly instantiate
        
            


    @property
    def tracking_run(self) -> str:
        return self.run_key.run

    @property
    def tracking_epoch(self) -> str:
        return self.last_storage_key.epoch

    @property
    def tracking_step(self) -> str:
        return self.last_storage_key.step

"""

Experiment Key: <CORE_NAME, VERSION, BACKBONE, QUALIFIER>
Run         <run>
Storage Key <epoch-step>

"""