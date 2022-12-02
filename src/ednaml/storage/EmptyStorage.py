from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, ExperimentKey, KeyMethods, StorageArtifactType


class EmptyStorage(BaseStorage):
    def apply(self, storage_url: str, **kwargs):
        self.run = -1
        self.storage_name: str = kwargs.get("storage_name")
        self.experiment_key: ExperimentKey = kwargs.get("experiment_key")
        self.storage_path = ""
        self.run_dir = ""

    def path_of_artifact(self, *args, **kwargs):
        return ""

    def canonical_path_of_artifact(self, *args, **kwargs):
        return ""

    def download(
        self, ers_key: ERSKey, destination_file_name: str, canonical: bool = False
    ) -> bool:
        return False

    def upload(
        self, source_file_name: str, ers_key: ERSKey, canonical: bool = False
    ) -> bool:
        return False

    def getLatestStorageKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:

        return_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        return_key.storage.epoch = -1
        return_key.storage.step = -1
        return return_key

    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        return self.run

    def getLatestEpochOfArtifact(self, ers_key: ERSKey) -> ERSKey:
        return None

    def getLatestStepOfArtifactWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        return None

    def getKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:
        return None

    def checkEpoch(self, ers_key: ERSKey) -> ERSKey:
        return None

    def checkStep(self, ers_key: ERSKey) -> ERSKey:
        return None

    def setTrackingRun(self, tracking_run: int) -> None:
        self.run = tracking_run
