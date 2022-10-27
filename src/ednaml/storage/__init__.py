from ednaml.config.SaveConfig import SaveConfig
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.storage.AzureStorage import AzureStorage
from ednaml.utils.SaveMetadata import SaveMetadata



class StorageManager:
    def __init__(self, saveMetadata: SaveMetadata, saveOptions: SaveConfig):
        self.metadata = saveMetadata
        self.saveoptions = saveOptions

    
