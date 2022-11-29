from typing import List
from glob import glob
import os, shutil
import re, warnings
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, KeyMethods, StorageArtifactType

# TODO take care of runs...
class LocalStorage(BaseStorage):
    def apply(self, storage_url: str, **kwargs):
        # storage_url is the desination directory
        os.makedirs(storage_url, exist_ok=True)
        self.storage_directory = "%s-v%s-%s-%s" % self.experiment_key.getKey()
        self.storage_path = os.path.join(storage_url, self.storage_directory)
        self.run_dir: str = None
        # TODO This is where Runs come into effect, perhaps...
        # For runs, we need to integrate this into StorageManager
        os.makedirs(self.storage_path, exist_ok=True)
        self.file_basename = kwargs.get("file_basename", "_".join([self.experiment_key.model_core_name, 
                                                                        "v%s"%self.experiment_key.model_version,
                                                                        self.experiment_key.model_backbone,
                                                                        self.experiment_key.model_qualifier]))

        self.path_ends = {
            StorageArtifactType.MODEL: "_model.pth",
            StorageArtifactType.ARTIFACT: "_artifact.pth",
            StorageArtifactType.PLUGIN: "_plugin.pth",
            StorageArtifactType.METRIC: "_metric.json",
            StorageArtifactType.CONFIG: "_config.yml",
            StorageArtifactType.LOG: ".log",
        }

    def path_of_artifact(self, epoch: int, step: int, artifact: StorageArtifactType) -> os.PathLike:
        return "_".join([self.file_basename, "epoch"+str(epoch),"step"+str(step)]) + self.path_ends[artifact]

    def setTrackingRun(self, tracking_run: int):
        
        self.run_dir = str(tracking_run)
        os.makedirs(os.path.join(self.storage_path, self.run_dir), exist_ok=True)


    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        """Return the maximum run for this Storage with the saved ExperimentKey. if no run exist, return -1.

        Args:
            artifact (StorageArtifactType, optional): _description_. Defaults to None.

        Returns:
            int: _description_
        """
        rundirs = [int(item.name) for item in os.scandir(self.storage_path) if not item.name.startswith(".")]
        if artifact is None:
            if len(rundirs) == 0:
                return -1
            else:
                return max(rundirs)
        else:
            raise NotImplementedError()


    def upload(self, source_file_name: str, ers_key: ERSKey):
        if os.path.exists(source_file_name):
            shutil.copy2(source_file_name, 
                os.path.join(self.storage_path, self.run_dir, self.path_of_artifact(epoch=ers_key.storage.epoch, step=ers_key.storage.step, artifact=ers_key.storage.artifact)))
            return True
        return False

    def download(self, ers_key: ERSKey, destination_file_name: str) -> bool:
        if self.getKey(ers_key=ers_key) is not None:
            shutil.copy2(os.path.join(self.storage_path, self.run_dir, self.path_of_artifact(epoch=ers_key.storage.epoch, step=ers_key.storage.step, artifact=ers_key.storage.artifact)), 
                destination_file_name)
            return True
        return False
        

    def getLatestStepOfArtifactWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        artifact_paths = os.path.join(self.storage_path, self.run_dir,  "*"+self.path_ends[ers_key.storage.artifact])
        artifact_list = glob(artifact_paths)
        artifact_basenames = [os.path.basename(item) for item in artifact_list]
        _re = re.compile(r".*epoch([0-9]+)_step([0-9]+)%s"%(self.path_ends[ers_key.storage.artifact].replace(".", "\.")))
        max_step = [int(item[2]) for item in [_re.search(item) for item in artifact_basenames] if int(item[1]) == ers_key.storage.epoch]

        if len(max_step) == 0:
            return None
        else:
            ers_key.storage.step = max(max_step)
            return ers_key

    def getKey(self, ers_key: ERSKey) -> ERSKey:
        if os.path.exists(os.path.join(self.storage_path, self.run_dir, self.path_of_artifact(epoch=ers_key.storage.epoch, step=ers_key.storage.step, artifact=ers_key.storage.artifact))):
            return ers_key
        return None

    def checkEpoch(self, ers_key: ERSKey) -> ERSKey:
        all_epochs = self._getAllEpochs(ers_key=ers_key)
        if ers_key.storage.epoch in all_epochs:
            return ers_key
        return None

    def checkStep(self, ers_key: ERSKey) -> ERSKey:
        return self.getKey(ers_key=ers_key)

    def getLatestEpochOfArtifact(self, ers_key: ERSKey) -> ERSKey:
        max_epoch = self._getAllEpochs(ers_key=ers_key)
        if len(max_epoch) == 0:
            return None
        else:
            ers_key.storage.epoch = max(max_epoch)
            return ers_key

    def _getAllEpochs(self, ers_key: ERSKey) -> List[int]:
        ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        # TODO modify or fix this in case of errors...?
        artifact_paths = os.path.join(self.storage_path, self.run_dir,  "*"+self.path_ends[ers_key.storage.artifact])
        artifact_list = glob(artifact_paths)
        artifact_basenames = [os.path.basename(item) for item in artifact_list]
        _re = re.compile(r".*epoch([0-9]+)_step([0-9]+)%s"%(self.path_ends[ers_key.storage.artifact].replace(".", "\.")))
        max_epoch = [int(item[1]) for item in [_re.search(item) for item in artifact_basenames]]
        return max_epoch


    def getLatestStorageKey(self, ers_key: ERSKey) -> ERSKey: # TODO need to adjust how files are saved so we can extract storagekey regardless of artifact type
        """Get the latest StorageKey in this Storage, given ERSKey with provided ExperimentKey, 
        RunKey, and Artifact in StorageKey.

        Args:
            ers_key (ERSKey): _description_
        """
        if ers_key.storage.artifact is not StorageArtifactType.MODEL:
            warnings.warn("`getLatestStorageKey` is not supported for artifacts other than MODEL for LocalStorage")

        model_paths = os.path.join(self.storage_path, self.run_dir,  "*model.pth")
        model_list = glob(model_paths)
        model_basenames = [os.path.basename(item) for item in model_list]
        _re = re.compile(r".*epoch([0-9]+)_step([0-9]+)_model\.pth")
        compiled = [_re.search(item) for item in model_basenames]
        max_epoch = [int(item[1]) for item in compiled]
        
        ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        if len(max_epoch) == 0:
            ers_key.storage.epoch = -1
            ers_key.storage.step = -1
        else:
            ers_key.storage.epoch = max(max_epoch)
            max_step = [int(item[2]) for item in compiled if int(item[1]) == ers_key.storage.epoch]
            ers_key.storage.step = max(max_step)
        return ers_key
