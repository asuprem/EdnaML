from glob import glob
import os, shutil
from typing import Union
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ExperimentKey, StorageArtifactType, StorageNameStruct
from ednaml.utils.SaveMetadata import SaveMetadata

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
                            "v%i"%self.experiment_key.model_version,
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


    def upload(self, source_file_name: str, file_struct: StorageNameStruct):
        shutil.copy2(source_file_name, 
            os.path.join(self.storage_path, self.convert_struct_to_file(file_struct)))

    def download(self, file_struct: StorageNameStruct, destination_file_name: str):
        shutil.copy2(os.path.join(self.storage_path, self.convert_struct_to_file(file_struct)), 
            destination_file_name)


    def convert_struct_to_file(self, file_struct: StorageNameStruct) -> Union[str, os.PathLike]:
        return self.path_ends[file_struct.artifact_type]((file_struct.run, file_struct.epoch, file_struct.step))