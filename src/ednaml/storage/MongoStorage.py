from typing import List
from glob import glob
import os, shutil
import re, warnings
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, KeyMethods, StorageArtifactType



class MongoStorage(BaseStorage):
    """MongoStorage saves to a mongo-db specified in arguments. Username and password can be passed directly into the arguments, or with enviroment variables.

    The following artifacts are supported for MongoStorage:
        Configuration
        Metrics

    The following artifact are not supported directly, but may be supported by logging a SaveRecord through Metrics
        Log: log files cannot be stored efficiently with MongoDB. In future, we will add options to record where logs should be upload so that timestamps and URLs can be recorded in Mongo
        Plugins: plugin files cannot be stored with MongoDB. In future, we will add connectors, e.g. Azure or LocalStorage, so that the backend storage can record where plugins are stored
        Models: See plugins
        Training Artifacts: See plugins
        Code: See plugins
        Extras: TBD


    Args:
        BaseStorage (_type_): _description_
    """
    def apply(self, storage_url: str, **kwargs):
        from pymongo import MongoClient
        # Set up client
        self.mc = MongoClient(
            **kwargs.get("mongo_client")
        )

        # Set up database
        self.db_name= kwargs.get("db_name", "ednaml")
        self.db = self.mc[self.db_name]
        
        # Set up collections if they do not already exist
        # Set up experiments collection
        self.experiments = self.db["experiments"]
        # Set up config collection
        self.configs = self.db["configs"]
        # set up metrics collection
        self.metrics = self.db["metrics"]
        # set up logs collection (eventually)
        self.logs = self.db["logs"]


        self.experiment_id = self._addExperimentKey()


    def _addExperimentKey(self):
        response = self.experiments.insert_one(
            self.experiment_key.todict()
        )
        return response.inserted_id



    def upload(self, source_file_name: str, ers_key: ERSKey, canonical: bool = False) -> bool:
        # So, in canonical, storage and epoch are -1, indicating default version, maybe...?
        # plus canonical is true
        # later, we can add a constraint that if canonical is true, sorage and epoch = -1, and if storage and epoch = -1, canonical = True

        if ers_key.storage.artifact == StorageArtifactType.CONFIG:
            self.uploadConfig(source_file_name=source_file_name, ers_key=ers_key, canonical=canonical)
        elif ers_key.storage.artifact == StorageArtifactType.METRIC:
            self.uploadMetric(source_file_name=source_file_name, ers_key=ers_key, canonical=canonical)
        else:
            self.log("Uploading {artifact} is not supported by MongoStorage"%ers_key.storage.artifact.value)



    def uploadConfig(self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False):
        return super().uploadConfig(source_file_name, ers_key, canonical)


    def uploadMetric(self, source_file_name: os.PathLike, ers_key: ERSKey, canonical: bool = False):
        return super().uploadMetric(source_file_name, ers_key, canonical)

        