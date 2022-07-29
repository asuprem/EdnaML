from typing import Dict
from azure.storage.blob import BlockBlobService
from azure.storage.blob import AppendBlobService
from ednaml.storage.BaseStorage import BaseStorage
import os,shutil,gzip

class AzureStorage(BaseStorage):        
    def build_params(self, **kwargs):
        self.storage_account_key = kwargs.get("storage_account_key")
        self.storage_account = kwargs.get("storage_account", "")
        self.containername = kwargs.get("containername", "default")
        self.blobname = kwargs.get("blobname", "default")
        self.local_azfilename = kwargs.get("local_filename", self.blobname) # "../Data/cars_datasets_models2"
        self.uploadblobname = kwargs.get("uploadblobname","default")    # TODO this needs to be dealt with w.r.t. config saves...

    def read(self):
        blob_service_client_instance = BlockBlobService(account_name=self.storage_account, 
            account_key=self.storage_account_key)

        blob_service_client_instance.create_container(self.containername)
        downloadpath = os.path.join("./",self.blobname)

        print("\nDownloading blob to " + downloadpath)
        blob_service_client_instance.get_blob_to_path( self.containername, self.blobname, downloadpath)
        if not os.path.exists(self.local_azfilename): 
            with gzip.open(downloadpath, 'rb') as f_in: 
                with open(self.local_azfilename, 'wb') as f_out: 
                    shutil.copyfileobj(f_in, f_out) 

    def write(self, data):  # TODO should upload name be specified here as well...????
        #print("Storage args = ", self.STORAGE_ARGS)
        blob_service_client_instance = BlockBlobService(account_name=self.storage_account, 
            credential=self.storage_account_key)

        blob_client_instance_upload = blob_service_client_instance.get_blob_client(
            self.containername, self.uploadblobname, snapshot=None)
        
        blob_client_instance_upload.upload_blob(data)   # TODO Possibly save data from somewhere. Is it a stream...need to read up on this

        #blob_client_instance_upload.append_blob_from_text()

    def append(self,data):
        print("This is upload code in Azure")
        append_blob_service = AppendBlobService(
            account_name=self.storage_account, 
            account_key=self.storage_account_key)
        # Creates an append blob for this app.
        # What does this do...?
        append_blob_service.create_container( self.containername)
        append_blob_service.create_blob( self.containername, 'log.txt')
        append_blob_service.append_blob_from_text( self.containername, 'log.txt', data)

    def copy(self,src):
        print("Uploading/copying logs to cloud storage")    
        with open(src, 'r') as content_file:
            content = content_file.read()
            self.append(content)
