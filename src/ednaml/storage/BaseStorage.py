from os import PathLike
from ednaml.utils.SaveMetadata import SaveMetadata


class BaseStorage:
    storage_type: str
    storage_url: str
    index: str
    run: str
    def __init__(self, type, url, **kwargs):
        self.storage_type = type
        self.storage_url = url
        self.build_params(**kwargs)
        


    def build_params(self, **kwargs):
        pass

    def read(self):
        print("Base read call")

    def write(self, data):
        print("Base write call",data)

    def append(self,data):
        print("Append call",data)

    def copy(self,src):
        print("Copy call ",src)

    def load(self, index: str, run: str, file_name: str, local_path: PathLike):
        """This loads a file from the index-run-file_name triplet. Specifically, it copies that file into the local file path

        Args:
            index (str): _description_
            run (str): _description_
            file_name (str): _description_
            local_path (PathLike): _description_
        """
       
        pass

    def save(self, index: str, run: str, file_path: PathLike):
        """This saves a provided file, such as config, model, model-artifact, plugin, or metrics file.

        Load takes an index-run-file_name triplet to identify where the provided file should be stored.
            - <mnist-resnet18-all-v1>-<run1>-<config.yml>
            - <mnist-resnet18-all-v1>-<run1>-<model.pth>
            - <mnist-resnet18-all-v1>-<run1>-<model-artifacts.pth>
            - <mnist-resnet18-all-v1>-<run1>-<metrics.json>

        A Storage class can save however it wants to. For example, the built-in mongo class 
        converts the config.ymls into mongo inputs, the metrics.json into mongo inputs,
        and the model.pth uploaded a mongo file

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

    def setRun(self, run):
        pass

    def getRun(self, run):
        pass
