from ednaml.config.EdnaMLConfig import EdnaMLConfig
from typing import Dict
#from azure.storage.blob import BlobServiceClient
#from azure.storage.blob import AppendBlobService
from ednaml.config.SaveConfig import SaveConfig

class BaseStorage:
    TYPE: str
    STORAGE_ARGS: Dict
    URL: str
    #SaveConfig: type[EdnaMLConfig.STORAGE] #directly reffering to original class.
    #config: EdnaMLConfig

    def __init__(self, type, storage_args, url, **kwargs):    # diff between : and = 
        self.TYPE = type
        self.STORAGE_ARGS = storage_args
        self.URL = url
        #self.SaveConfig = configs.STORAGE

    def read(self):
        print("Base read call")

    def write(self, data):
        print("Base write call",data)

    def append(self,data):
        print("Append call",data)

    def copy(self,src):
        print("Copy call ",src)