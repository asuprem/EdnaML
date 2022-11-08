
from ednaml.utils import ExperimentKey, StorageArtifactType, StorageNameStruct
from ednaml.utils.SaveMetadata import SaveMetadata

class BaseStorage:
    storage_name: str
    storage_url: str
    experiment_key: ExperimentKey
    def __init__(self, experiment_key: ExperimentKey, storage_name, storage_url, **storage_kwargs):
        self.storage_name = storage_name
        self.storage_url = storage_url
        self.experiment_key = experiment_key
        self.apply(self.storage_url, **storage_kwargs)
        
    def apply(self, save_metadata: SaveMetadata, storage_url: str, **kwargs):
        """Builds the internal state of the Storage module

        Args:
            url (str): The URL for the storage endpoint

        Kwargs:
            As neeeded
        """
        raise NotImplementedError()

    def download(self, file_struct: StorageNameStruct, destination_file_name: str):
        """Use the storage backend to download a file with the `file_struct` key into a destination file

        Args:
            file_struct (StorageNameStruct): Key of file to retrieve
            destination_file_name (str): Destination file name to save retrieved file in
        """
        raise NotImplementedError()

    def upload(self, source_file_name: str, file_struct: StorageNameStruct):
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
