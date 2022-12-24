import logging, os
from typing import Callable, Dict, List, Union
from ednaml.config.BackupOptionsConfig import BackupOptionsConfig
from typing import TYPE_CHECKING

from ednaml.storage.BaseStorage import BaseStorage
from ednaml.storage.LocalStorage import LocalStorage
from ednaml.storage.EmptyStorage import EmptyStorage
from ednaml.utils import StorageArtifactType, ExperimentKey, RunKey, StorageKey, ERSKey
if TYPE_CHECKING:
    from ednaml.config.EdnaMLConfig import EdnaMLConfig


class StorageManager:
    """StorageManager is a helper class for storage-related tasks in EdnaML

    It has 4 tasks:
    1. Provide a ERSKey given the epoch, run, step, and type parameters
    2. Provide a local-file-name given a StorageNameStruct
    3. Provide a trigger-mechanism for checking upload requirements given the configuration file and current epoch and step
    4. Provide the storage name for a given upload type (e.g. config, logfile, model, etc)
    """

    logger: logging.Logger
    _experiment_key: ExperimentKey
    storage_manager_mode: str  # loose | download | strict. Loose: If storage is defined, we upload/download. Download: we download if storage is defined. Strict: We upload/download ONLY if allowed
    storage_trigger_mode: str  # loose | strict
    storage_manager_strict: bool
    storage_trigger_strict: bool
    storage_mode: str
    backup_mode: str
    cfg: 'EdnaMLConfig'
    run_key: RunKey
    latest_storage_key: StorageKey
    local_save_directory: str
    local_storage: LocalStorage
    artifact_references: Dict[StorageArtifactType, BackupOptionsConfig]
    epoch_triggers: Dict[StorageArtifactType, Callable[[int], bool]]
    step_triggers: Dict[StorageArtifactType, Callable[[int], bool]]

    def __init__(
        self,
        logger: logging.Logger,
        cfg: 'EdnaMLConfig',
        experiment_key: ExperimentKey,
        storage_trigger_mode: str = "loose",  # Literal["loose", "strict"]                # Trigger mode determines how often we check whether we should upload
        storage_manager_mode: str = "loose",  # Literal["loose", "strict", "download_only"]                # Manager mode determines whether StorageManager will check performBackup before downloading or uploading
        storage_mode: str = "local",  # Literal["local", "empty"]                # Whether to save locally or not
        backup_mode: str = "hybrid",
        backup_mode_canonical: List[str] = ["log", "config", "plugin", "metric", "code"]
    ):  # Literal["canonical", "ers", "hybrid", "custom"],
        """_summary_

        Args:
            logger (logging.Logger): _description_
            cfg (EdnaMLConfig): _description_
            experiment_key (ExperimentKey): _description_
            storage_trigger_mode (str, optional): _description_. Defaults to "loose".
            storage_manager_mode (str, optional): _description_. Defaults to "loose".
            storage_mode (str, optional): _description_. Defaults to "loose".

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        self.logger = logger

        self.experiment_key = experiment_key
        self.storage_manager_mode = self.validate(
            "storage_manager_mode",
            ["strict", "loose", "download_only"],
            storage_manager_mode,
        )
        self.storage_trigger_mode = self.validate(
            "storage_trigger_mode", ["strict", "loose"], storage_trigger_mode
        )
        self.storage_mode = self.validate(
            "storage_mode", ["empty", "local"], storage_mode
        )
        self.backup_mode = self.validate(
            "backup_mode", ["canonical", "ers", "hybrid", "custom"], backup_mode
        )

        self.log("Initializing StorageManager")
        self.log(
            "\tusing experiment_key:      \t{ekey}".format(
                ekey=self.experiment_key.getExperimentName()
            )
        )
        self.log(
            "\twith storage_manager_mode: \t{mode}".format(
                mode=self.storage_manager_mode
            )
        )
        self.log(
            "\twith storage_trigger_mode: \t{mode}".format(
                mode=self.storage_trigger_mode
            )
        )
        self.log("\twith storage_mode:         \t{mode}".format(mode=self.storage_mode))
        self.log("\twith backup_mode:          \t{mode}".format(mode=self.backup_mode))
        self.cfg = cfg

        self.run_key = None
        self.latest_storage_key = None
        # Add options here for where to save local files, i.e. not directly in ./
        if storage_mode == "local":
            self.local_save_directory = "%s-v%s-%s-%s" % self.experiment_key.getKey()
            self.log("\tUsing local save directory: \t%s" % self.local_save_directory)
            os.makedirs(self.local_save_directory, exist_ok=True)
        else:
            self.local_save_directory = ""
            self.log("\tUsing empty ephemeral storage")

        self.file_basename = "experiment"
        self.log("\tUsing file basename: \t%s" % self.file_basename)

        if self.storage_mode == "local":
            self.local_storage = LocalStorage(
                logger=self.logger,
                experiment_key=self.experiment_key,
                storage_name="ednaml-local-storage-reserved",
                storage_url="./",
                file_basename=self.file_basename,
            )
            self.log("Generated `ednaml-local-storage-reserved` LocalStorage object")
        else:
            self.local_storage = EmptyStorage(
                experiment_key=self.experiment_key,
                storage_name="ednaml-empty-storage-reserved",
                storage_url="./",
            )
            self.log(
                "Generated empty `ednaml-empty-storage-reserved` LocalStorage object"
            )
            self.log("WARNING: Nothing will be saved.")
        # We create a fast, O(1) reference for each of the save-options to avoid switch statements later
        self.artifact_references = {
            StorageArtifactType.MODEL: self.cfg.SAVE.MODEL_BACKUP,
            StorageArtifactType.ARTIFACT: self.cfg.SAVE.ARTIFACTS_BACKUP,
            StorageArtifactType.PLUGIN: self.cfg.SAVE.PLUGIN_BACKUP,
            StorageArtifactType.METRIC: self.cfg.SAVE.METRICS_BACKUP,
            StorageArtifactType.CONFIG: self.cfg.SAVE.CONFIG_BACKUP,
            StorageArtifactType.LOG: self.cfg.SAVE.LOG_BACKUP,
        }

        if self.backup_mode == "ers":
            canonical = []
            ers = [item for item in self.artifact_references]
        elif self.backup_mode == "canonical":
            canonical = [item for item in self.artifact_references]
            ers = []
        elif self.backup_mode == "hybrid":
            ers = [StorageArtifactType.MODEL, StorageArtifactType.ARTIFACT]
            canonical = [item for item in self.artifact_references if item not in ers]
        elif self.backup_mode == "custom":
            canonical = [StorageArtifactType(canonical_item) for canonical_item in backup_mode_canonical]
            ers = [item for item in self.artifact_references if item not in canonical]
        else:
            raise NotImplementedError()
        self.backup_canonical_references = {
            item: True if item in canonical else False
            for item in self.artifact_references
        }

        class LooseTriggerMethod:
            def __init__(self, trigger_frequency: int, initial_state: int = -1, base=0):
                self.trigger_frequency: int = trigger_frequency
                self.state = initial_state
                self.base = base

            def __call__(self, check_value: int) -> bool:
                # Get the quotient, i.e. whether we are at a multiplicative factor of the trigger_frequency
                cv = int(check_value / self.trigger_frequency)
                if cv != self.state:  # Check if same multiplicative factor as before
                    self.state = cv
                    if (
                        cv >= self.base
                    ):  # Check if we are above the lowest threshold. 0 for EPOCH, 1 for STEP.
                        return True
                return False

        # Trigger methods tell us whether to trigger an upload or not. To improve speed, we cache many of the parameters here.
        if self.storage_trigger_mode == "strict":
            # We create a fast O(1) reference to the trigger checking methods here for epochs
            self.epoch_triggers = {
                StorageArtifactType.MODEL: (lambda x: False)
                if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.ARTIFACT: (lambda x: False)
                if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.PLUGIN: (lambda x: False)
                if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.METRIC: (lambda x: False)
                if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.CONFIG: (lambda x: False)
                if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0)),
                StorageArtifactType.LOG: (lambda x: False)
                if self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0
                else (lambda x: (x % self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0)),
            }
            self.local_epoch_trigger = (lambda x: False) if self.cfg.SAVE.SAVE_FREQUENCY == 0 else (lambda x: (x % self.cfg.SAVE.SAVE_FREQUENCY == 0))

        elif self.storage_trigger_mode == "loose":
            self.epoch_triggers = {
                StorageArtifactType.MODEL: (lambda x: False)
                if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.MODEL_BACKUP.FREQUENCY),
                StorageArtifactType.ARTIFACT: (lambda x: False)
                if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY),
                StorageArtifactType.PLUGIN: (lambda x: False)
                if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY),
                StorageArtifactType.METRIC: (lambda x: False)
                if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.METRICS_BACKUP.FREQUENCY),
                StorageArtifactType.CONFIG: (lambda x: False)
                if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY),
                StorageArtifactType.LOG: (lambda x: False)
                if self.cfg.SAVE.LOG_BACKUP.FREQUENCY == 0
                else LooseTriggerMethod(self.cfg.SAVE.LOG_BACKUP.FREQUENCY),
            }
            self.local_epoch_trigger = (lambda x: False) if self.cfg.SAVE.SAVE_FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.SAVE_FREQUENCY)
        else:
            raise NotImplementedError()

        self.log("Generated EpochTrigger checks")
        if self.storage_trigger_mode == "strict":
            # We create a fast O(1) reference to the trigger checking methods here for epochs
            self.step_triggers = {
                StorageArtifactType.MODEL: (lambda x: False)
                if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0
                else (lambda x: (x % self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.ARTIFACT: (lambda x: False)
                if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0
                else (
                    lambda x: (x % self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0)
                ),
                StorageArtifactType.PLUGIN: (lambda x: False)
                if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0
                else (lambda x: (x % self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.METRIC: (lambda x: False)
                if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0
                else (lambda x: (x % self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.CONFIG: (lambda x: False)
                if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0
                else (lambda x: (x % self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0)),
                StorageArtifactType.LOG: (lambda x: False)
                if self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0
                else (lambda x: (x % self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0)),
            }
            self.local_step_trigger = (lambda x: False) if self.cfg.SAVE.STEP_SAVE_FREQUENCY == 0 else (lambda x: (x % self.cfg.SAVE.STEP_SAVE_FREQUENCY == 0))

        elif self.storage_trigger_mode == "loose":
            self.step_triggers = {
                StorageArtifactType.MODEL: (lambda x: False)
                if self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.MODEL_BACKUP.FREQUENCY_STEP, base=1
                ),
                StorageArtifactType.ARTIFACT: (lambda x: False)
                if self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.ARTIFACTS_BACKUP.FREQUENCY_STEP, base=1
                ),
                StorageArtifactType.PLUGIN: (lambda x: False)
                if self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.PLUGIN_BACKUP.FREQUENCY_STEP, base=1
                ),
                StorageArtifactType.METRIC: (lambda x: False)
                if self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.METRICS_BACKUP.FREQUENCY_STEP, base=1
                ),
                StorageArtifactType.CONFIG: (lambda x: False)
                if self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.CONFIG_BACKUP.FREQUENCY_STEP, base=1
                ),
                StorageArtifactType.LOG: (lambda x: False)
                if self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP == 0
                else LooseTriggerMethod(
                    self.cfg.SAVE.LOG_BACKUP.FREQUENCY_STEP, base=1
                ),
            }
            self.local_step_trigger = (lambda x: False) if self.cfg.SAVE.STEP_SAVE_FREQUENCY == 0 else LooseTriggerMethod(self.cfg.SAVE.STEP_SAVE_FREQUENCY, base=1)
        else:
            raise NotImplementedError()
        self.log("Generated StepTrigger checks")

    def log(self, msg: str) -> None:
        """Logs a message to the logger with a `[StorageManager]` prefix

        Args:
            msg (str): Message to log
        """
        self.logger.debug("[StorageManager] %s" % msg)

    def getERSKey(
        self,
        epoch: int,
        step: int,
        artifact_type: StorageArtifactType = StorageArtifactType.MODEL,
    ) -> ERSKey:
        """Combine the current  `experiment_key` and `run_key` with the provided epoch, step, and artifact to generate
        an ERSKey.

        Args:
            epoch (int): Epoch value for ERSKey
            step (int): Step value for ERSKey
            artifact_type (StorageArtifactType, optional): The StorageArtifactType for the ERSKey. Defaults to StorageArtifactType.MODEL.

        Returns:
            ERSKey: A complete ERSKey
        """
        return ERSKey(
            self.experiment_key, self.run_key, StorageKey(epoch, step, artifact_type)
        )

    @property
    def experiment_key(self) -> ExperimentKey:
        """Returns the current `experiment_key`

        Returns:
            ExperimentKey: The current `experiment_key`
        """
        return self._experiment_key

    @experiment_key.setter
    def experiment_key(self, ekey: ExperimentKey):
        """Sets the current `experiment_key`

        Args:
            ekey (ExperimentKey): The current experiment key
        """
        self._experiment_key = ekey

    def getRunKey(self) -> RunKey:
        """Gets the current `run_key`

        Returns:
            RunKey: The current `run_key`
        """
        return self.run_key

    def getLatestStorageKey(self) -> StorageKey:
        """Gets the latest StorageKey tracked by the StorageManager. The latest Storage Key refers to the most recent artifact created in either local or remote storage. Usually, the MODEL artifact is used as the reference artifact to compute the latest Storage Key.

        Returns:
            StorageKey: The latest Storage Key tracked by this StorageManager
        """
        return self.latest_storage_key

    def getLatestStepOfArtifactWithEpoch(
        self,
        storage: Dict[str, BaseStorage],
        epoch: int = None,
        ers_key: ERSKey = None,
        artifact: StorageArtifactType = StorageArtifactType.MODEL,
    ) -> Union[ERSKey, None]:
        """Given an epoch or an ERSKey with epoch value, return ERSKey of the
        latest Artifact in remote OR local. Returns `None` if there is no ERSKey.

        If `storage_manager_mode` is 'strict', the remote is checked only if
        backup is allowed for the artifact. If `storage_manager_mode` is 'loose'
        or 'download_only', then both the local and remote storages are checked
        regardless of whether backup is allowed, as long as a remote storage is
        provided.

        Note: `storage_manager_mode` defaults to 'strict' for EML and
        'download_only' for ED.

        Args:
            epoch (int, optional): The epoch value to find the latest
                step for. Used if `ers_key` is not provided. Defaults to None.
            ers_key (ERSKey, optional): An ERSKey whose epoch value
                will be used. Defaults to None.
            artifact (StorageArtifactType, optional): The artifact to use as
                reference. Defaults to StorageArtifactType.MODEL.
        """
        if ers_key is None:
            ers_key = self.getERSKey(epoch=epoch, step=0, artifact_type=artifact)
        remote_step = self.getERSKey(epoch=-1, step=-1, artifact_type=artifact)
        if self.performBackup(artifact_type=artifact) or (
            not self.storage_manager_strict
        ):
            storage_name = self.getStorageNameForArtifact(artifact_type=artifact)
            if storage_name not in storage:
                remote_step = None
            else:
                remote_step = storage[storage_name].getLatestStepOfArtifactWithEpoch(ers_key=ers_key)
        local_step = self.local_storage.getLatestStepOfArtifactWithEpoch(
            ers_key=ers_key
        )
        if remote_step is None:
            return local_step  # Could be none or greater...
        if local_step is None:
            return remote_step
        if remote_step.storage.step > local_step.storage.step:
            return remote_step
        return local_step  # if remote step equal or less than local

    def getLatestEpochOfArtifact(
        self,
        storage: Dict[str, BaseStorage],
        ers_key: ERSKey = None,
        artifact: StorageArtifactType = StorageArtifactType.MODEL,
    ) -> ERSKey:
        if ers_key is None:
            ers_key = self.getERSKey(epoch=0, step=0, artifact_type=artifact)
        remote_epoch = self.getERSKey(epoch=-1, step=-1, artifact_type=artifact)
        if self.performBackup(artifact_type=artifact) or (
            not self.storage_manager_strict
        ):
            storage_name = self.getStorageNameForArtifact(artifact_type=artifact)
            if storage_name not in storage:
                remote_epoch = None
            else:
                remote_epoch = storage[storage_name].getLatestEpochOfArtifact(ers_key=ers_key)
        local_epoch = self.local_storage.getLatestEpochOfArtifact(ers_key=ers_key)
        if remote_epoch is None:
            return local_epoch  # Could be None or Greater
        if local_epoch is None:
            return remote_epoch
        if remote_epoch.storage.epoch > local_epoch.storage.epoch:
            return remote_epoch
        return local_epoch  # if remote step equal or less than local

    def checkEpoch(
        self,
        storage: Dict[str, BaseStorage],
        epoch: int = None,
        ers_key: ERSKey = None,
        artifact: StorageArtifactType = None,
    ) -> bool:
        """Check if an epoch exists in either remote or local storage

        Args:
            storage (Dict[str, BaseStorage]): _description_
            epoch (int, optional): _description_. Defaults to None.
            ers_key (ERSKey, optional): _description_. Defaults to None.
            artifact (StorageArtifactType, optional): _description_. Defaults to None.
        """
        if ers_key is None:
            ers_key = self.getERSKey(epoch=epoch, step=0, artifact_type=artifact)
        remote_response = None
        if self.performBackup(artifact_type=artifact) or (
            not self.storage_manager_strict
        ):
            storage_name = self.getStorageNameForArtifact(artifact_type=artifact)
            if storage_name not in storage:
                remote_response = None
            else:
                remote_response = storage[storage_name].checkEpoch(ers_key=ers_key)
        local_response = self.local_storage.checkEpoch(ers_key=ers_key)

        if (remote_response is not None) or (local_response is not None):
            return True
        return False

    def checkStep(
        self,
        storage: Dict[str, BaseStorage],
        epoch: int = None,
        step: int = None,
        ers_key: ERSKey = None,
        artifact: StorageArtifactType = None,
    ):
        if ers_key is None:
            ers_key = self.getERSKey(epoch=epoch, step=step, artifact_type=artifact)
        remote_response = None
        if self.performBackup(artifact_type=artifact) or (
            not self.storage_manager_strict
        ):
            storage_name = self.getStorageNameForArtifact(artifact_type=artifact)
            if storage_name not in storage:
                remote_response = None
            else:
                remote_response = storage[storage_name].checkStep(ers_key=ers_key)
        local_response = self.local_storage.checkStep(ers_key=ers_key)

        if (remote_response is not None) or (local_response is not None):
            return True
        return False

    def getLatestERSKey(
        self, artifact: StorageArtifactType = StorageArtifactType.MODEL
    ) -> ERSKey:
        return self.getERSKey(
            epoch=self.latest_storage_key.epoch,
            step=self.latest_storage_key.step,
            artifact_type=artifact,
        )

    def getNextERSKey(
        self, artifact: StorageArtifactType = StorageArtifactType.MODEL
    ) -> ERSKey:
        return self.getERSKey(
            epoch=self.latest_storage_key.epoch + 1, step=0, artifact_type=artifact
        )

    def download(self, storage_dict: Dict[str, BaseStorage], ers_key: ERSKey) -> bool:
        """Download the file(s) corresponding to the ERSKey if they do not already exist locally.

        Args:
            storage_dict (Dict[str, BaseStorage]): _description_
            ers_key (ERSKey): _description_
        """

        local_path = self.getLocalSavePath(ers_key=ers_key)
        storage_name = self.getStorageNameForArtifact(ers_key.storage.artifact)

        if not os.path.exists(local_path):
            if self.storage_mode == "empty":
                self.log(
                    "Not downloading ERSKey `{key}` from storage {storage} due to EmptyStorage".format(
                        key=ers_key.printKey(), storage=storage_name
                    )
                )
                return False
            else:
                if self.storage_manager_strict and not self.performBackup(
                    ers_key.storage.artifact
                ):
                    self.log(
                        "Not downloading ERSKey `{key}` from storage {storage} due to strict checking".format(
                            key=ers_key.printKey(), storage=storage_name
                        )
                    )
                    return False

                if storage_name not in storage_dict:
                    self.log(
                        "Attempted downloading ERSKey `{key}` but Storage {storage} does not exist".format(
                            key=ers_key.printKey(), storage=storage_name
                        )
                    )
                    return False
                self.log(
                    "Downloading ERSKey `{key}` from storage {storage} into {path}".format(
                        key=ers_key.printKey(), storage=storage_name, path=local_path
                    )
                )
                return storage_dict[storage_name].download(
                    ers_key=ers_key,
                    destination_file_name=local_path,
                    canonical=self.backup_canonical_references[
                        ers_key.storage.artifact
                    ],
                )
        else:
            self.log(
                "ERSKey `{key}` already exists locally. Skipping.".format(
                    key=ers_key.printKey()
                )
            )
        return True  # Already exists.

    def upload(self, storage_dict: Dict[str, BaseStorage], ers_key: ERSKey) -> bool:
        """Upload the file(s) corresponding to the ERSKey. If they already exist, Storage will throw an error.

        Args:
            storage_dict (Dict[str, BaseStorage]): _description_
            ers_key (ERSKey): _description_
        """
        source_file_name = self.getLocalSavePath(ers_key=ers_key)
        storage_name = self.getStorageNameForArtifact(ers_key.storage.artifact)

        if self.storage_mode == "empty":
            self.log(
                "Not uploading ERSKey `{key}` due to EmptyStorage".format(
                    key=ers_key.printKey(), storage=storage_name
                )
            )
            return False
        if os.path.exists(source_file_name):
            if (
                self.storage_manager_strict
                or self.storage_manager_mode == "download_only"
            ) and not self.performBackup(ers_key.storage.artifact):
                self.log(
                    "Not uploading ERSKey `{key}` to storage {storage} due to strict checking or download_only mode".format(
                        key=ers_key.printKey(), storage=storage_name
                    )
                )
                return False
            if storage_name not in storage_dict:
                self.log(
                    "Attempted uploading ERSKey `{key}` but Storage {storage} does not exist".format(
                        key=ers_key.printKey(), storage=storage_name
                    )
                )
                return False
            self.log(
                "Uploading {path} into storage {storage}, with ERSKey `{key}`".format(
                    key=ers_key.printKey(), storage=storage_name, path=source_file_name
                )
            )
            storage_dict[storage_name].upload(
                ers_key=ers_key,
                source_file_name=source_file_name,
                canonical=self.backup_canonical_references[ers_key.storage.artifact],
            )
            return True
        else:
            self.log(
                "Could not find any file for ERSKey `{key}`, at local path {path}".format(
                    key=ers_key.printKey(), path=source_file_name
                )
            )
        return False  # Could not upload due to file not found

    def getLocalFileName(self, ers_key: ERSKey) -> Union[str, os.PathLike]:
        """Creates the local file name for the provided StorageKey. This does not contain experiment details, or run details.

        Args:
            storage_struct (StorageNameStruct): The StorageKey to construct a local file name for

        Returns:
            Union[str,os.PathLike]: The constructed file name that combines attributes of the StorageKey
        """
        return self.local_storage.path_of_artifact(
            ers_key.storage.epoch, ers_key.storage.step, ers_key.storage.artifact
        )

    def getLocalSavePath(self, ers_key: ERSKey) -> Union[str, os.PathLike]:
        """Provides the complete path to the file name for the provided StorageKey

        Args:
            ers_key (ERSKey): _description_

        Returns:
            Union[str, os.PathLike]: _description_
        """
        return os.path.join(
            self.local_storage.storage_path,
            self.local_storage.run_dir,
            self.getLocalFileName(ers_key=ers_key),
        )

    def getUploadTriggerForEpoch(
        self, epoch: int, artifact_type: StorageArtifactType
    ) -> bool:
        """Determines whether an upload should occur at this epoch given the
        provided StorageArtifactType.

        Args:
            epoch (int): Epoch number
            artifact_type (StorageArtifactType): The artifact type for this trigger check.

        Returns:
            bool: If true, we should upload. If false, we should not upload.
        """
        return self.epoch_triggers[artifact_type](epoch)

    def getUploadTriggerForStep(
        self, step: int, artifact_type: StorageArtifactType
    ) -> bool:
        return self.step_triggers[artifact_type](step)

    def getSaveTriggerForEpoch(self, epoch: int) -> bool:
        return self.local_epoch_trigger(epoch)
    def getSaveTriggerForStep(self ,step: int) -> bool:
        return self.local_step_trigger(step)
    


    def performBackup(self, artifact_type: StorageArtifactType) -> bool:
        return self.artifact_references[artifact_type].BACKUP

    def getStorageNameForArtifact(self, artifact_type: StorageArtifactType) -> str:
        return self.artifact_references[artifact_type].STORAGE_NAME

    def setTrackingRun(
        self,
        storage_dict: Dict[str, BaseStorage] = None,
        tracking_run: int = None,
        new_run: bool = False,
    ) -> None:
        # NOTE: We check remote tracking run if backup is allowed OR if we are in download_only/loose mode!!!
        if tracking_run is None:
            self.log("Searching for tracking run with `new_run`: %s" % str(new_run))
            max_run_list = [
                self._getMaximumRun(storage_dict, self.getStorageNameForArtifact(artifact_key))
                if (
                    self.performBackup(artifact_key)
                    or (not self.storage_manager_strict)
                )
                else -1
                for artifact_key in self.artifact_references
            ]
            max_run = max(max_run_list)
            if max_run < 0:
                tracking_run = 0
            else:
                tracking_run = max_run + int(new_run)
            self.log("Remote tracking run is %i" % tracking_run)

            local_tracking_run = self.local_storage.getMaximumRun() + int(new_run)
            self.log("Local tracking run is %i" % tracking_run)
            tracking_run = max(tracking_run, local_tracking_run)
        else:
            self.log("Using provided `tracking_run`: %s" % str(tracking_run))

        # NOTE at this time, we ignore all this complication, and just save the config in the run directly.
        # Storage's uploadConfig handles doubles by renaming the existing config by including the most recent StorageKey from saved model(s)
        # Then, the provided config is uploaded
        self._setTrackingRun(storage_dict, tracking_run)

        # ers_key = self.getERSKey(epoch = 0, step = 0, artifact_type=StorageArtifactType.CONFIG)
        # self.cfg.save(self.getLocalSavePath(ers_key=ers_key))
        # storage_dict[self.getStorageNameForArtifact(StorageArtifactType.CONFIG)].uploadConfig(ers_key = ers_key,
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

    def _getMaximumRun(self, storage_dict: Dict[str, BaseStorage], storage_name, ):
        if storage_name in storage_dict:
            return storage_dict[storage_name].getMaximumRun()
        return -1

    def _setTrackingRun(
        self, storage_dict: Dict[str, BaseStorage], tracking_run: int
    ) -> None:
        self.run_key = RunKey(run=tracking_run)
        self.log(
            "Tracking run in `ednaml-local-storage-reserved` set to: %i" % tracking_run
        )
        # Create the run directory locally.
        self.local_storage.setTrackingRun(tracking_run)

        # Inform the storages that we have a run:
        for artifact_key in self.artifact_references:
            if (
                self.performBackup(artifact_type=artifact_key)
                or not self.storage_manager_strict
            ):
                storage_name = self.getStorageNameForArtifact(artifact_key)
                if storage_name not in storage_dict:
                    self.log(
                        "Did not set tracking run for Artifact `%s` at Storage `%s`. Storage %s does not exist."
                        % (artifact_key.value, storage_name, storage_name)
                    )
                else:
                    storage_dict[storage_name].setTrackingRun(tracking_run)
                    self.log(
                        "Tracking run for Artifact `%s` at Storage `%s` set to: %i"
                        % (artifact_key.value, storage_name, tracking_run)
                    )

        # Copy the files over??? Or save that for other methods to perform...

    def setLatestStorageKey(
        self, storage_dict: Dict[str, BaseStorage], artifact: StorageArtifactType = None
    ) -> None:
        """Check local storage as well as remote storage(s), if backup is performed on a remote storage, for the latest epoch-step pair when something was saved.
        
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
        final_ers_key = self.searchLatestERSKey(
            storage_dict=storage_dict, artifact=artifact
        )
        self.latest_storage_key = StorageKey(
            epoch=final_ers_key.storage.epoch,
            step=final_ers_key.storage.step,
            artifact=artifact,
        )

    def searchLatestERSKey(
        self, storage_dict: Dict[str, BaseStorage], artifact: StorageArtifactType = None
    ) -> ERSKey:
        self.log(
            "Intializing reference StorageKey to (-1,-1), with reference artifact: %s"
            % artifact.value
        )
        ers_key: ERSKey = self.getERSKey(epoch=-1, step=-1, artifact_type=artifact)
        if (
            self.performBackup(artifact_type=artifact)
            or not self.storage_manager_strict
        ):
            # 
            storage_name = self.getStorageNameForArtifact(artifact_type=artifact)
            if storage_name not in storage_dict:
                self.log(
                    "Not searching remote. Storage %s does not exist." % storage_name
                )
                remote_ers_key = None
            else:
                if not self.storage_manager_strict:
                    self.log(
                        "Retrieving remote ERSKey because `storage_manager_mode` is not `strict`"
                    )
                else:
                    self.log("Retrieving remote ERSKey because backup is allowed")
                if self.backup_canonical_references[artifact]:
                    remote_ers_key: ERSKey = storage_dict[
                        storage_name
                    ].getLatestStorageKey(
                        ers_key, canonical=True
                    )
                else:
                    remote_ers_key: ERSKey = storage_dict[
                        storage_name
                    ].getLatestStorageKey(ers_key)
        else:
            remote_ers_key = None
        local_ers_key: ERSKey = self.local_storage.getLatestStorageKey(ers_key=ers_key)

        if remote_ers_key is None:
            self.log(
                "Did not find any remote ERSKey because backup is not enabled for reference artifact: %s"
                % (artifact.value)
            )
        else:
            self.log(
                "Found remote ERSKey `%s` with reference artifact: %s"
                % (remote_ers_key.printKey(), artifact.value)
            )
        self.log(
            "Found local ERSKey `%s` with reference artifact: %s"
            % (local_ers_key.printKey(), artifact.value)
        )

        final_ers_key = None
        if remote_ers_key is None:
            final_ers_key = local_ers_key
        else:
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

        self.log(
            "Obtained latest StorageKey at (%i,%i), with reference artifact: %s"
            % (final_ers_key.storage.epoch, final_ers_key.storage.step, artifact.value)
        )
        return final_ers_key

    def updateStorageKey(self, ers_key: ERSKey) -> None:
        self.latest_storage_key.epoch = ers_key.storage.epoch
        self.latest_storage_key.step = ers_key.storage.step

    @property
    def tracking_run(self) -> str:
        return self.run_key.run

    @property
    def tracking_epoch(self) -> str:
        return self.latest_storage_key.epoch

    @property
    def tracking_step(self) -> str:
        return self.latest_storage_key.step

    @property
    def storage_manager_mode(self) -> str:
        return self._storage_manager_mode

    @storage_manager_mode.setter
    def storage_manager_mode(self, mode: str):
        if mode not in ["loose", "strict", "download_only"]:
            raise ValueError(
                "`storage_manager_mode` must be one of [`loose`, `strict`], got %s"
                % (str(mode))
            )
        self._storage_manager_mode = mode
        self._storage_manager_mode_strict = (
            True if self._storage_manager_mode == "strict" else False
        )

    @property
    def storage_manager_strict(self) -> bool:
        return self._storage_manager_mode_strict

    @property
    def storage_trigger_mode(self) -> str:
        return self._storage_trigger_mode

    @storage_trigger_mode.setter
    def storage_trigger_mode(self, mode: str):
        if mode not in ["loose", "strict"]:
            raise ValueError(
                "`storage_trigger_mode` must be one of [`loose`, `strict`], got %s"
                % (str(mode))
            )
        self._storage_trigger_mode = mode
        self._storage_trigger_mode_strict = (
            True if self._storage_trigger_mode == "strict" else False
        )

    @property
    def storage_trigger_strict(self) -> bool:
        return self._storage_trigger_mode_strict

    def validate(self, var_name: str, var_options: List[str], var_value: str):
        if var_value not in var_options:
            raise ValueError(
                "Unsupported value for `{varname}` <{varvalue}>. Must be one of {varlist}".format(
                    varname=var_name,
                    varvalue=var_value,
                    varlist="["
                    + " ,".join(["`" + item + "`" for item in var_options])
                    + "]",
                )
            )
        return var_value


"""

Experiment Key: <CORE_NAME, VERSION, BACKBONE, QUALIFIER>
Run         <run>
Storage Key <epoch-step>

"""
