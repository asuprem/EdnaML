from logging import Logger
from os import PathLike
from ednaml.utils.SaveMetadata import SaveMetadata


class BaseStorage:
    """BaseStorage is the abstract class for all Storages in Edna. 

    Storages implement file upload, download, and querying. Storages are connected to backend data stores
    to backup experiment progress and deployment results (through the config files, model.pth, and metrics.json files)

    EdnaML and Deploy tasks have 6 components. Given an experiment name `[EXPNAME]`, files use the following
    naming convention:
        - Configuration     -- config.yml
        - Model             -- model-[EXPNAME]_epoch[#].pth
        - Model Artifacts   -- artifacts-[EXPNAME]_epoch[#].pth
        - Model Plugins     -- plugins-[EXPNAME].pth
        - Metrics           -- metrics.json
        - Log file          -- [EXPNAME]-logger.log

    A configuration file can specify any number of Storages for any subset of these components. As such, Storages should
    handle all 6 components. Custom Storages can choose to raise errors if they are provided a component they cannot handle.

    Attributes:
        storage_type: The Storage type is the Storage class. This is used during Storage construction, and saved inside the Storage.
        storage_url: Since most Storages use some form of resource locator, the `storage_url` records a generic URL. For example, a 
            LocalStorage might store the path to the LocalStorage folder, while a MongoStorage might store the Mongo host address + port.
        index: Each experiment has a string index for its name. Experiments are unique.
        run: Each experiment may have multiple runs under it. Runs are mononotically increasing integers starting with 0
        logger: Logger
    """
    storage_type: str
    storage_url: str
    index: str
    run: str
    logger: Logger
    def __init__(self, type, url, logger, **kwargs):
        """Sets up the BaseStorage and passes any keyword arguments to `build_params`, which can be implemented in Custom Storages
        Keyword arguments are from CONFIG->STORAGE->STORAGE_ARGS

        Args:
            type (_type_): _description_
            url (_type_): _description_
            logger (_type_): _description_
        """
        self.storage_type = type
        self.storage_url = url
        self.logger = logger
        self.build_params(**kwargs)
        


    def build_params(self, **kwargs):
        """Set up the internal parameters for a BaseStorage. Storages implement this with 
        """
        pass

    def load(self, index: str, run: str, file_name: str, local_path: PathLike):
        """Each of Edna 6 components (covered in __init__) exists as a file.
        
        Each file is uniquely identified by a triplet of <index, run, filename>. So, Edna will request file loads 
        with this triplet; as such, Storages should store files with this triplet as one form of identifier.
        
        For example, the model stored in the fifth epoch of an EdnaML experiment "CIFAR-CNN-v2" in the second run (runs
        start at 0) of this experiment would be requested as:  <CIFAR-CNN-v2, 1, model-CIFAR-CNN-v2_epoch5.pth>

        A Storage class can save files however it wants to. When requested, Storage must provide the original file
        to the `local_path` so that the EdnaML experiment can use it.
        
        Args:
            index (str): The index to fetch. If not provided, `load` should rely on the self.index variable
            run (str): The run to fetch. If not provided, `load` should rely on the self.run variable
            file_name (str): _description_
            local_path (PathLike): _description_
        """
       
        pass

    def save(self, index: str, run: str, file_path: PathLike):
        """This saves a provided file under the triplet <index, run, file_name> in the Storage backend.

        A Storage class can save however it wants to. For example, the built-in mongo class 
        converts the config.ymls into mongo inputs, the metrics.json into mongo inputs,
        and the model.pth uploads a mongo file. In each case, however, 

        The Mlflow version is more tightly integrated, and uses the Mlflow backend stores, if set up properly, to 
        store metrics, files, and configs.

        Args:
            index (str): _description_
            run (str): _description_
            file_name (str): _description_
        """
        pass
    
    def exists(self, index: str, run: str, file_path: PathLike):
        return False

    # These are necessary for any storage that interacts with config files, i.e. the tracking solutions 
    # Storages that are purely artifact store need not implement this, as it will not be called.
    def setIndex(self, index):
        pass

    def getMostRecentRun(self):
        pass

    def getNextRun(self):
        pass

    def setRun(self, run):
        pass

    def getRun(self, run):
        pass

    def query(self, index, run, file):
        pass


    def read(self):
        print("Base read call")

    def write(self, data):
        print("Base write call",data)

    def append(self,data):
        print("Append call",data)

    def copy(self,src):
        print("Copy call ",src)