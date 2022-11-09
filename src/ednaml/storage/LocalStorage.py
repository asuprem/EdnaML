from glob import glob
import os, shutil
import re
from typing import Union
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, ExperimentKey, StorageArtifactType, StorageNameStruct
from ednaml.utils.SaveMetadata import SaveMetadata

# TODO take care of runs...
class LocalStorage(BaseStorage):
    def apply(self, storage_url: str, **kwargs):
        # storage_url is the desination directory
        os.makedirs(storage_url, exist_ok=True)
        self.storage_directory = "%s-v%s-%s-%s" % (
            self.experiment_key.model_core_name,
            self.experiment_key.model_version,
            self.experiment_key.model_backbone,
            self.experiment_key.model_qualifier,
        )
        self.storage_path = os.path.join(storage_url, self.storage_directory)
        # TODO This is where Runs come into effect, perhaps...
        # For runs, we need to integrate this into StorageManager
        os.makedirs(self.storage_path, exist_ok=True)
        self.file_basename = "_".join([self.experiment_key.model_core_name, 
                            "v%s"%self.experiment_key.model_version,
                            self.experiment_key.model_backbone,
                            self.experiment_key.model_qualifier])

        self.path_ends = {
            StorageArtifactType.MODEL: lambda x : "_".join([self.file_basename,"run"+x[0],"epoch"+x[1],"step"+x[2],"model.pth"]),
            StorageArtifactType.ARTIFACT: lambda x : "_".join([self.file_basename,"run"+x[0],"epoch"+x[1],"step"+x[2],"artifact.pth"]),
            StorageArtifactType.PLUGIN: lambda x: "_".join([self.file_basename, "plugin.pth"]),
            StorageArtifactType.METRIC: lambda x: "".join([self.file_basename, ".json"]),
            StorageArtifactType.CONFIG: lambda x: "".join([self.file_basename, ".yml"]),
            StorageArtifactType.LOG: lambda x: "".join([self.file_basename, ".log"]),
        }

    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        """Return the maximum run for this Storage with the saved ExperimentKey

        Args:
            artifact (StorageArtifactType, optional): _description_. Defaults to None.

        Returns:
            int: _description_
        """
        rundirs = [int(item.name) for item in os.scandir(self.storage_path)]
        if artifact is None:
            if len(rundirs) == 0:
                return 0
            else:
                return max(rundirs)
        else:
            raise NotImplementedError()


    def upload(self, source_file_name: str, ers_key: ERSKey):
        shutil.copy2(source_file_name, 
            os.path.join(self.storage_path, self.convert_key_to_file(ers_key)))

    def download(self, ers_key: ERSKey, destination_file_name: str):
        shutil.copy2(os.path.join(self.storage_path, self.convert_key_to_file(ers_key)), 
            destination_file_name)
        

    def getLatestModelWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        model_paths = os.path.join(self.storage_path, self.path_ends[StorageArtifactType.MODEL]((ers_key.run.run, ers_key.storage.epoch, "*")))
        model_list = glob(model_paths)
        model_basenames = [os.path.basename(item) for item in model_list]
        _re = re.compile(r".*epoch([0-9]+)_step([0-9]+)_model\.pth")
        max_step = [int(item[2]) for item in [_re.search(item) for item in model_basenames]]

        if len(max_step) == 0:
            return None
        else:
            ers_key.storage.step = max(max_step)
            return ers_key

    def getKey(self, ers_key: ERSKey) -> ERSKey:
        if os.path.exists(self.convert_key_to_file(ers_key=ers_key)):
            return ers_key
        return None

    def getLatestModelEpoch(self, ers_key: ERSKey) -> ERSKey:
        # TODO modify or fix this in case of errors...?
        model_paths = os.path.join(self.storage_path, self.path_ends[StorageArtifactType.MODEL]((ers_key.run.run, "*", "*")))
        model_list = glob(model_paths)
        model_basenames = [os.path.basename(item) for item in model_list]
        _re = re.compile(r".*epoch([0-9]+)_step([0-9]+)_model\.pth")
        max_epoch = [int(item[1]) for item in [_re.search(item) for item in model_basenames]]

        if len(max_epoch) == 0:
            return None
        else:
            ers_key.storage.epoch = max(max_epoch)
            return ers_key



    def convert_key_to_file(self, ers_key: ERSKey) -> Union[str, os.PathLike]:
        return self.path_ends[ers_key.storage.artifact]((ers_key.run.run, ers_key.storage.epoch, ers_key.storage.step))