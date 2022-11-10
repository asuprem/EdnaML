import os
from ednaml.utils import ERSKey, ExperimentKey, StorageArtifactType

class BaseStorage:
    storage_name: str
    storage_url: str
    experiment_key: ExperimentKey
    def __init__(self, experiment_key: ExperimentKey, storage_name, storage_url, **storage_kwargs):
        self.storage_name = storage_name
        self.storage_url = storage_url
        self.experiment_key = experiment_key        
        
        self.apply(self.storage_url, **storage_kwargs)

        
    def apply(self, storage_url: str, **kwargs):
        """Builds the internal state of the Storage module

        Args:
            url (str): The URL for the storage endpoint

        Kwargs:
            As neeeded
        """
        raise NotImplementedError()

    def download(self, ers_key: ERSKey, destination_file_name: str):
        """Use the storage backend to download a file with the `file_struct` key into a destination file

        Args:
            file_struct (StorageNameStruct): Key of file to retrieve
            destination_file_name (str): Destination file name to save retrieved file in
        """
        raise NotImplementedError()

    def upload(self, source_file_name: str, ers_key: ERSKey):
        """Upload a local file into the storage backend

        Args:
            source_file_name (str): The file to upload
            file_struct (StorageNameStruct): The key for the file to upload
        """
        raise NotImplementedError()

        


    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        """Returns the maximum run for this Storage with the current `self.experiment_key`.

        If `artifact` is provided, return the maximum run that contains a saved `artifact`.

        If `artifact` is not provided, return the maximum run overall.

        Args:
            artifact (StorageArtifactType, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            int: _description_
        """
        raise NotImplementedError()

    def getLatestModelWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        """Returns the latest model's ERSKey created (using the step as the comparator) with the provided Epoch. If there is no model, return None.

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()

    def getLatestModelEpoch(self, ers_key: ERSKey) -> ERSKey:
        """Return the latest epoch of the model given ERS key. If no models exist, return None

        Args:
            ers_key (ERSKey): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()

    def getKey(self, ers_key: ERSKey) -> ERSKey:
        """Returns the key if it exists in the Storage, else return None

        Args:
            ers_key (ERSKey): _description_

        Returns:
            ERSKey: _description_
        """
        raise NotImplementedError()

    def uploadConfig(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadConfig(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)

    def uploadLog(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadLog(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)

    def uploadModel(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadModel(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)

    def uploadModelArtifact(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadModelArtifact(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)

    def uploadModelPlugin(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadModelPlugin(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)
    
    def uploadMetric(self, source_file_name: os.PathLike, ers_key: ERSKey):
        self.upload(source_file_name=source_file_name, ers_key=ers_key)

    def downloadMetric(self, ers_key: ERSKey, destination_file_name: os.PathLike):
        self.download(ers_key=ers_key, destination_file_name=destination_file_name)



    
