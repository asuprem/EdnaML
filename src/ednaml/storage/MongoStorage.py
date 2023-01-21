from time import time
from typing import List, Tuple, Union
from glob import glob
import os, shutil
import re, warnings
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, ExperimentKey, KeyMethods, RunKey, StorageArtifactType, StorageKey
try:
    from bson.objectid import ObjectId
except:
    ObjectId = None
import yaml

class MongoStorage(BaseStorage):
    """MongoStorage saves to a mongo-db specified in arguments. Username and password can be passed directly into the arguments, or with environment variables.

    The following artifacts are supported for MongoStorage:
        Configuration
        Metrics

    The following artifact are not supported directly, but may be supported in the future
        Log: log files cannot be stored efficiently with MongoDB. In future, we may add option to record where logs should be upload so that timestamps and URLs can be recorded in Mongo
        Plugins: Plugin files should be stored in a dedicated file store instead of Mongo. In future, we will add connectors, e.g. Azure or LocalStorage, so that the backend storage can record where plugins are stored
        Models: See plugins (since they are files)
        Training Artifacts: See plugins (since they are files)
        Code: See plugins (since they are files)
        Extras: TBD


    Args:
        BaseStorage (_type_): _description_
    """
    def set_save_record(self):
        self.save_record = True

    def apply(self, storage_url: str, **kwargs):
        """We set up a connection to the Mongo Backend.

        Args:
            storage_url (str): _description_
        """
        from pymongo import MongoClient
        # Set up client
        self.log("Initializing MongoClient")
        self.mc = MongoClient(
            host=self.storage_url,
            **kwargs.get("mongo_client")
        )

        # Set up database
        self.db_name= kwargs.get("db_name", "ednaml")
        self.log("Connecting to database {db_name}".format(db_name=self.db_name))
        self.db = self.mc[self.db_name]
        
        # Set up collections if they do not already exist

        # Set up experiments collection. Experiments collection tracks the experiments (by names) plus the runs. 
        # In future, we will add SaveRecords to this as well, so we can track ERSKeys here (the actual file or content associated with the key goes elsewhere)
        self.log("Setting up `experiments` collection")
        self.experiments = self.db["experiments"]
        # Set up runs collection
        self.log("Setting up `runs` collection")
        self.runs = self.db["runs"]
        # Set up config collection
        self.log("Setting up `configs` collection")
        self.configs = self.db["configs"]
        # set up metrics collection
        self.log("Setting up `metrics` collection")
        self.metrics = self.db["metrics"]
        # set up records collection
        self.log("Setting up `records` collection")
        self.records = self.db["records"]
        # set up logs collection (eventually)
        # self.logs = self.db["logs"]

        # Add the experiment key into the experiments collection and get the id
        self.local_cache = {"runs":{}, "experiments":{}, "configs":{}, "metrics":{}}
        self.experiment_id = self._addExperimentKey(self.experiment_key)


        self.supported_artifacts = {
            StorageArtifactType.MODEL: False,
            StorageArtifactType.ARTIFACT: False,
            StorageArtifactType.PLUGIN: False,
            StorageArtifactType.LOG: False,
            StorageArtifactType.METRIC: True, # <--------
            StorageArtifactType.CONFIG: True, # <--------
            StorageArtifactType.CODE: False,
            StorageArtifactType.EXTRAS: False,
        }

        
        # Only supported artifacts here.
        self.collection_reference = {
            StorageArtifactType.CONFIG: self.configs,
            StorageArtifactType.METRIC: self.metrics
        }


    def _addExperimentKey(self, experiment_key: ExperimentKey) -> ObjectId:
        experiment_exists = self.experiments.find_one(experiment_key.todict())
        if experiment_exists is None:
            self.log("Inserting experiment key {ekey}".format(ekey=experiment_key.getExperimentName()))
            response = self.experiments.insert_one(
                experiment_key.todict()
            )
            self.local_cache["experiments"][experiment_key.getKey()] = response.inserted_id
            return response.inserted_id
        else:
            self.log("Experiment key {ekey} already exists".format(ekey=experiment_key.getExperimentName()))
            self.local_cache["experiments"][experiment_key.getKey()] = experiment_exists["_id"]
            return experiment_exists["_id"]

    def setTrackingRun(self, tracking_run: int) -> None:
        self.log("Setting tracking run to {run}".format(run=tracking_run))
        self.run_id = self._addRun(tracking_run)
        self._setTrackingRun(tracking_run=tracking_run)

    def _addRun(self, tracking_run: int) -> ObjectId:
        run_exists = self.runs.find_one(
            {
                "experiment": self.experiment_id,
                "run": tracking_run
            }
        )
        if run_exists is None:
            response = self.runs.insert_one(
                {
                    "experiment": self.experiment_id,
                    "run": tracking_run
                }
            )
            self.local_cache["runs"][tracking_run] = response.inserted_id
            return response.inserted_id
        else:
            self.local_cache["runs"][tracking_run] = run_exists["_id"]
            return run_exists["_id"]

    def getKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:
        
        # We first get the id for the provided experiment key. 
        # Then we get the id for the provided run key
        # Then, for the artifact, we call the respective artifact checker with the given experiment and run id to see if there is one entry...

        if self.supported_artifacts[ers_key.storage.artifact]:
            # Check whether the experiment key exists in the local cache (i.e. we have retrieved it before)
            experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
            if run_id is None:
                return None

            # Now we check if the storage key exists. We have already filtered out the artifact bit.
            query_document = self._construct_ers_query(experiment_id, run_id, ers_key.storage, canonical)

            ers_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document)
            if ers_response is None:
                return None
            else:
                return ers_key
        return None

    def _getExperimentIdIfExists(self, experiment_key: ExperimentKey) -> Union[ObjectId, None]:
        cache_check_id = self.local_cache["experiments"].get(experiment_key.getKey(), None)
        if cache_check_id is None:
            # Now we query the remote.
            remote_response = self.experiments.find_one(experiment_key.todict())
            if remote_response is None:
                return None # The E of ERS does not exist
            else:
                # Get the experiment id, save it in cache for future
                experiment_id = remote_response["_id"]
                self.local_cache["experiments"][experiment_key.getKey()] = experiment_id
        else:
            experiment_id = cache_check_id
        return experiment_id

    def _getRunIdIfExists(self, experiment_id: ObjectId, run_key: RunKey) -> Union[ObjectId, None]:
        run_cache_check_id = self.local_cache["runs"].get(run_key.run, None)
        if run_cache_check_id is None:
            # Now we query the remote.
            remote_response = self.runs.find_one({
                "experiment": experiment_id,
                "run": run_key.run
            })
            if remote_response is None:
                return None # The R of ERS does not exist
            else:
                # Get the run id, save it in cache for future
                run_id = remote_response["_id"]
                self.local_cache["runs"][run_key.run] = run_id
        else:
            run_id = run_cache_check_id
        return run_id
    
    def _getERIdIfExists(self, ers_key: ERSKey) -> Union[Tuple[ObjectId, ObjectId], None]:
        experiment_id = self._getExperimentIdIfExists(experiment_key=ers_key.experiment)
        if experiment_id is not None:
            run_id = self._getRunIdIfExists(experiment_id, ers_key.run)
            if run_id is not None:
                return (experiment_id, run_id)
        return None, None

    def _construct_ers_query(self, experiment_id: ObjectId, run_id: ObjectId, storage_key: StorageKey, canonical: bool):
        if canonical:
            return {
                "experiment": experiment_id,
                "run": run_id,
                "canonical": True
            }
        return {
                "experiment": experiment_id,
                "run": run_id,
                "canonical": False,
                "epoch": storage_key.epoch,
                "step": storage_key.step
        }

    def _construct_epoch_query(self, experiment_id: ObjectId, run_id: ObjectId, epoch: int, canonical: bool):
        if canonical:
            return {
                "experiment": experiment_id,
                "run": run_id,
                "canonical": True
            }
        return {
                "experiment": experiment_id,
                "run": run_id,
                "canonical": False,
                "epoch": epoch,
        }

    def _construct_er_query(self, experiment_id: ObjectId, run_id: ObjectId, canonical: bool):
        if canonical:
            return {
                "experiment": experiment_id,
                "run": run_id,
                "canonical": True
            }
        return  {
                "experiment": experiment_id,
                "run": run_id
                }



    def checkStep(self, ers_key: ERSKey) -> ERSKey:
        return self.getKey(ers_key=ers_key, canonical=False)

    def checkEpoch(self, ers_key: ERSKey) -> ERSKey:
        if self.supported_artifacts[ers_key.storage.artifact]:
            experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
            if run_id is None:
                return None
            query_document = self._construct_epoch_query(experiment_id=experiment_id, run_id=run_id, epoch=ers_key.storage.epoch, canonical=False)
            epoch_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document)
            
            if epoch_response is None:
                return None
            return ers_key
        return None


    def getLatestEpochOfArtifact(self, ers_key: ERSKey) -> ERSKey:
        if self.supported_artifacts[ers_key.storage.artifact]:
            experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
            if run_id is None:
                return None
            query_document = self._construct_er_query(experiment_id=experiment_id, run_id=run_id, canonical=False)
            epoch_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document, sort=[("epoch", -1)])
            
            if epoch_response is None:
                return None
            return epoch_response["epoch"]
        return None


    def getLatestStepOfArtifactWithEpoch(self, ers_key: ERSKey) -> ERSKey:
        if self.supported_artifacts[ers_key.storage.artifact]:
            experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
            if run_id is None:
                return None
            query_document = self._construct_epoch_query(experiment_id=experiment_id, run_id=run_id, epoch=ers_key.storage.epoch, canonical=False)
            step_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document, sort=[("step", -1)])
            
            if step_response is None:
                return None
            return step_response["step"]
        return None

    def _construct_maxrun_aggregate_query(self, experiment_id: ObjectId, reference_run_collection: str = "runs"):
        return [{"$match":{"experiment": experiment_id}},
                            {"$group":{"_id": "$run"}},
                            {"$lookup":{"from": reference_run_collection,"localField": "_id","foreignField": "_id","as": "_id"}},
                            {"$addFields":{"_id": "$_id.run"}},
                            {"$unwind": "$_id"},
                            {"$sort":{"_id": -1}},
                            {"$limit": 1}] 

    def getMaximumRun(self, artifact: StorageArtifactType = None) -> int:
        experiment_id = self._getExperimentIdIfExists(experiment_key=self.experiment_key)
        if experiment_id is None:
            raise KeyError("Expected `experiment` with key %s in MongoStorage. Could not find."%str(self.experiment_key.getKey()))
        if artifact is None:
            # Now we find the id of the document with maximum run value in self.runs
            # TODO we can probably delete above and directly use self.experiment_id, but I don't want race conditions until I verify a few things...
            query_document = {
                "experiment": experiment_id
            }
            run_response = self.runs.find_one(query_document, sort=[("run", -1)])
            if run_response is None:
                return -1
            return run_response["run"]

        else:
            # We check artifact-specific collection
            # We aleady have experiment id reference
            # We will obtain all run-ids within artifact-specific collection that have the correct experiment id
            # back reference to get the runs from the run ids
            # get the maximum run from that...
            aggregate_pipeline = self._construct_maxrun_aggregate_query(experiment_id=experiment_id, reference_run_collection="runs")
            run_response = self.collection_reference[artifact].aggregate(aggregate_pipeline)

            run_value = next(run_response,None)
            if run_value is None:
                return -1
            return run_value["_id"]

    def getLatestStorageKey(self, ers_key: ERSKey, canonical: bool = False) -> ERSKey:
        if ers_key.storage.artifact is not StorageArtifactType.MODEL:
            warnings.warn(
                "`getLatestStorageKey` is not supported for artifacts other than MODEL for MongoStorage"
            )

        # So, we need to find the latest storage key for the provided artifact...
        # If canonical, we need to find if anything exists with the canonical flag...

        experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
        absent_ers_key = KeyMethods.cloneERSKey(ers_key=ers_key)
        absent_ers_key.storage.epoch = -1
        absent_ers_key.storage.step = -1
        if run_id is None:
            return absent_ers_key
        
        # Now that we have the ids, we can check if the provided ERSKey exists
        if canonical:
            # Check for artifact in canonical mode

            query_document = self._construct_ers_query(experiment_id=experiment_id, run_id=run_id, storage_key=ers_key.storage, canonical = True)
            ers_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document)
            if ers_response is None:
                return absent_ers_key
            else:
                return ers_key

        else:
            # Here, we need to search for the latest storage key.
            query_document = self._construct_er_query(experiment_id=experiment_id, run_id=run_id, canonical = False)
            ers_response = self.collection_reference[ers_key.storage.artifact].find_one(query_document, sort=[("epoch", -1, "step", -1)])
            if ers_response is None:
                return absent_ers_key
            else:
                absent_ers_key.storage.epoch = ers_response["epoch"]
                absent_ers_key.storage.step = ers_response["step"]
                return absent_ers_key

    def upload_impl(self, source_file_name: str, ers_key: ERSKey, canonical: bool = False) -> bool:
        self.log("`upload_impl` is not meant to be called. Something has gone wrong.")

    def download_impl(self, ers_key: ERSKey, destination_file_name: str, canonical: bool = False) -> bool:
        self.log("`download_impl` is not meant to be called. Something has gone wrong.")

    
    def downloadConfig(self, ers_key: ERSKey, destination_file_name: Union[str, os.PathLike], canonical: bool = False):
        
        experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
        if run_id is None:
            return False

        # Now that we have the relevant details, we will attempt to download the specific configuration to local.
        query_document = self._construct_ers_query(experiment_id=experiment_id, run_id=run_id, storage_key=ers_key.storage, canonical = canonical)

        response = self.configs.find_one(query_document)
        if response is None:
            return False
        
        # Here, response is a document representing the configuration structure in its entirety.
        with open(destination_file_name, "w") as cfile:
            yaml.dump(response, cfile)
        return True

    # TODO //////////////////////////////////////////////////////////////////////////////////////////////////////////
    def uploadConfig(self, source_file_name: Union[str, os.PathLike], ers_key: ERSKey, canonical: bool = False):
        self.log("Config backup requested with ers_key {ers_key}".format(ers_key=ers_key.printKey()))
        experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
        if run_id is None:
            raise KeyError("Expected `experiment` with key %s in MongoStorage. Could not find."%str(ers_key.experiment.getKey()))
        if not os.path.exists(source_file_name):
            return False

        config_document = None
        with open(source_file_name, "r") as s_file:
            config_document = yaml.safe_load(s_file.read().strip())

        

        config_document["experiment"] = self.experiment_id
        config_document["run"] = self.run_id
        config_document["epoch"] = ers_key.storage.epoch
        config_document["step"] = ers_key.storage.step
        config_document["canonical"] = canonical
        
        response = self.configs.insert_one(config_document)
        if response.acknowledged:
            return True
        return False

    # We do not download metrics...maybe...
    # Unless we use the SaveRecord and download all metrics between SaveRecords...
    # But, for now, we DO NOT download metrics...
    def downloadMetric(self, ers_key: ERSKey, destination_file_name: Union[str, os.PathLike], canonical: bool = False):
        return False

    def uploadMetric(self, source_file_name: Union[str, os.PathLike], ers_key: ERSKey, canonical: bool = False):
        self.log("Metrics backup requested with ers_key {ers_key}".format(ers_key=ers_key.printKey()))
        experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
        if run_id is None:
            raise KeyError("Expected `experiment` with key %s in MongoStorage. Could not find."%str(ers_key.experiment.getKey()))
        if not os.path.exists(source_file_name):
            return False

        metrics_document = None
        metric_items = []
        with open(source_file_name, "r") as metrics_file:
            for line in metrics_file:
                metric_items.append(line.strip().split(","))

        #      0              1           2           3      4        5
        #  (metric_name, metric_type, metric_class, epoch, step, metric_value)
        metrics_document = [
            {
                "experiment": self.experiment_id,
                "run": self.run_id,
                "epoch": int(metric_item[3]),
                "step": int(metric_item[4]),
                "name": metric_item[0],
                "type": metric_item[1],
                "class": metric_item[2],
                "value": metric_item[5],
            }
            for metric_item in metric_items
        ]
        self.log("Backing up {number} metrics to {ers_key}".format(number = str(len(metric_items)), ers_key=ers_key.printKey()))
        response = self.metrics.insert_many(metrics_document)
        if response.acknowledged:
            return True
        return False


    def _recordArtifactSave(self, epoch: int, step: int, artifact: StorageArtifactType, storage_name: str, storage_class: str, storage_url: str, experiment_key: ExperimentKey, run: int) -> bool:
        self.log("Generating SaveRecord for {artifact} at epoch-step <{epoch}-{step}>".format(
            artifact = artifact.value,
            epoch = epoch,
            step = step
        ))
        ers_key = ERSKey(
            experiment=experiment_key,
            run=run,
            storage=StorageKey(epoch=-1,step=-1,artifact=StorageArtifactType.MODEL)
        )
        experiment_id, run_id = self._getERIdIfExists(ers_key=ers_key)
        if run_id is None:
            raise KeyError("Expected `experiment` with key %s in MongoStorage. Could not find."%str(ers_key.experiment.getKey()))
        
        save_record_document = {
                "experiment": self.experiment_id,
                "run": self.run_id,
                "epoch": epoch,
                "step": step,
                "artifact": artifact.value,
                "storage_name": storage_name,
                "storage_class": storage_class,
                "storage_url": storage_url
        }
        response = self.records.insert_one(save_record_document)
        if response.acknowledged:
            return True
        return False
        
    def log(self, msg):
        self.logger.info("[MongoStorage]" + msg)