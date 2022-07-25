from ednaml.config.EdnaMLConfig import EdnaMLConfig
from typing import Dict
from azure.storage.blob import BlockBlobService
from azure.storage.blob import AppendBlobService
from ednaml.config.SaveConfig import SaveConfig
from ednaml.storage.BaseStorage import BaseStorage
import os,shutil,gzip

class AzureStorage(BaseStorage):
    TYPE: str
    STORAGE_ARGS: Dict
    URL: str

    '''def __init__(self, configs: EdnaMLConfig, **kwargs):    # diff between : and = 
        self.TYPE = configs.TYPE
        self.STORAGE_ARGS = configs.STORAGE_ARGS
        self.URL = configs.URL
        #self.SaveConfig = configs.STORAGE'''

    def read(self):
        blob_service_client_instance = BlockBlobService(account_name=self.STORAGE_ARGS.get("storage_account", ""), 
            account_key=self.STORAGE_ARGS.get("storage_account_key"))

        blob_service_client_instance.create_container(self.STORAGE_ARGS.get("containername","default"))
        downloadpath = os.path.join("./",self.STORAGE_ARGS.get("blobname","default"))

        print("\nDownloading blob to " + downloadpath)
        blob_service_client_instance.get_blob_to_path( self.STORAGE_ARGS.get("containername","default"), self.STORAGE_ARGS.get("blobname","default"), downloadpath)
        az_jsonfile = "../Data/cars_datasets_models2"
        if not os.path.exists(az_jsonfile): 
            with gzip.open(downloadpath, 'rb') as f_in: 
                with open(az_jsonfile, 'wb') as f_out: 
                    shutil.copyfileobj(f_in, f_out) 

    def write(self, data):
        print("Storage args = ", self.STORAGE_ARGS)
        blob_service_client_instance = BlockBlobService(account_name=self.STORAGE_ARGS.get("storage_account", ""), 
            credential=self.STORAGE_ARGS.get("storage_account_key", ""))

        blob_client_instance_upload = blob_service_client_instance.get_blob_client(
            self.STORAGE_ARGS.get("containername","default"), self.STORAGE_ARGS.get("uploadblobname","default"), snapshot=None)
        
        blob_client_instance_upload.upload_blob(data)

        #blob_client_instance_upload.append_blob_from_text()

    def append(self,data):
        print("This is upload code in Azure")
        append_blob_service = AppendBlobService(
            account_name=self.STORAGE_ARGS.get("storage_account", ""), 
            account_key=self.STORAGE_ARGS.get("storage_account_key", ""))
        # Creates an append blob for this app.
        append_blob_service.create_container( self.STORAGE_ARGS.get("containername","default"))
        append_blob_service.create_blob( self.STORAGE_ARGS.get("containername","default"), 'log.txt')
        append_blob_service.append_blob_from_text( self.STORAGE_ARGS.get("containername","default"), 'log.txt', data)

    def copy(self,src):
        print("Uploading/copying logs to cloud storage")    
        with open(src, 'r') as content_file:
            content = content_file.read()
            self.append(content)