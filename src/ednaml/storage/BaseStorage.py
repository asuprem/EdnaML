from abc import ABC, abstractmethod
import logging
import os
from ednaml.utils import ERSKey, ExperimentKey, StorageArtifactType, StorageKey


class BaseStorage(ABC):
    storage_name: str
    storage_url: str
    experiment_key: ExperimentKey
    logger: logging.Logger

    def __init__(
        self, logger: logging.Logger, experiment_key: ExperimentKey, storage_name, storage_url, **storage_kwargs
    ):
        self.storage_name = storage_name
        self.storage_url = storage_url
        self.experiment_key = experiment_key
        self.logger = logger
        self.apply(self.storage_url, **storage_kwargs)

    @abstractmethod
    def apply(self, storage_url: str, **kwargs):
        """Builds the internal state of the Storage module

        Args:
            url (str): The URL for the storage endpoint

        Kwargs:
            As neeeded
        """
        raise NotImplementedError()

    @abstractmethod
    def download(
        self, ers_key: ERSKey, destination_file_name: str, canonical: bool = False
    ) -> bool:
        """Use the storage backend to download a file with the `file_struct` key into a destination file

        Args:
            file_struct (StorageNameStruct): Key of file to retrieve
            destination_file_name (str): Destination file name to save retrieved file in
        """
        raise NotImplementedError()
    @abstractmethod
    def upload(
        self, source_file_name: str, ers_key: ERSKey, canonical: bool = False
    ) -> bool:
        """Upload a local file into the storage backend

        Args:
            source_file_name (str): The file to upload
            file_struct (StorageNameStruct): The key for the file to upload
        """
        raise NotImplementedError()
    @abstractmethod
    def getLatestStorageKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:
        """Get the latest StorageKey in this Storage, given ERSKey with provided ExperimentKey,
        RunKey, and Artifact. If there is no latest StorageKey, return an ERSKey with -1 for epoch and storage.

        If canonical is set, check if a canonical file exists and return the provided ers_key as it is, or return -1/-1.

        Args:
            ers_key (ERSKey): _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        """Returns the maximum run for this Storage with the current `self.experiment_key`.

        If `artifact` is provided, return the maximum run that contains that specific `artifact`.

        If `artifact` is not provided, return the maximum run overall.

        If there is no run, return -1

        Args:
            artifact (StorageArtifactType, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            int: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def getLatestStepOfArtifactWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        """Returns the latest artifact's ERSKey created (using the step as the comparator)
        with the provided Epoch. If there is no artifact, return None.

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def getLatestEpochOfArtifact(self, ers_key: ERSKey) -> ERSKey:
        """Return the latest epoch of the given artifact in ERS key. If no models exist, return None

        Args:
            ers_key (ERSKey): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def getKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:
        """Returns the key if it exists in the Storage, else return None.

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def checkEpoch(self, ers_key: ERSKey) -> ERSKey:
        """Returns the key if the epoch exists in the Storage, else return None

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def checkStep(self, ers_key: ERSKey) -> ERSKey:
        """Returns the key if epoch AND step exists in the Storage, else return None

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def setTrackingRun(self, tracking_run: int) -> None:
        """Set the current run for this experiment.

        Args:
            tracking_run (int): The run for this experiment

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()
    @abstractmethod
    def uploadConfig(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadConfig(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )
    @abstractmethod
    def uploadLog(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadLog(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )
    @abstractmethod
    def uploadModel(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadModel(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )
    @abstractmethod
    def uploadModelArtifact(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadModelArtifact(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )
    @abstractmethod
    def uploadModelPlugin(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadModelPlugin(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )
    @abstractmethod
    def uploadMetric(
        self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False
    ):
        return self.upload(
            source_file_name=source_file_name, ers_key=ers_key, canonical=canonical
        )
    @abstractmethod
    def downloadMetric(
        self,
        ers_key: ERSKey,
        destination_file_name: os.PathLike,
        canonical: bool = False,
    ):
        return self.download(
            ers_key=ers_key,
            destination_file_name=destination_file_name,
            canonical=canonical,
        )


    def log(self, msg):
        self.logger.info("[BaseStorage]" + msg)