from ednaml.config.EdnaMLConfig import EdnaMLConfig
from typing import Dict
from azure.storage.blob import BlockBlobService
from azure.storage.blob import AppendBlobService
from ednaml.config.SaveConfig import SaveConfig
from ednaml.storage.BaseStorage import BaseStorage
import os,shutil,gzip

class LocalStorage(BaseStorage):
    def read(self):
        print("Implement download file here")
        downloadpath = "This is dummy name"
        az_jsonfile = "../Data/cars_datasets_models2"
        if not os.path.exists(az_jsonfile): 
            with gzip.open(downloadpath, 'rb') as f_in: 
                with open(az_jsonfile, 'wb') as f_out: 
                    shutil.copyfileobj(f_in, f_out) 

    def write(self, data):
        print("Implement write file here")

        #blob_client_instance_upload.append_blob_from_text()

    def append(self,data):
        print("Implement append file here")
    def copy(self,src,dst):
        print("Implement copy to storage here")
