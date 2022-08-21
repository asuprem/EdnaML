from glob import glob
from ednaml.config.EdnaMLConfig import EdnaMLConfig
from typing import Dict
from ednaml.config.SaveConfig import SaveConfig
from ednaml.storage.BaseStorage import BaseStorage
import os,shutil,gzip

class LocalStorage(BaseStorage):
    # storage_url has the path where we will backup our files
    def build_params(self, **kwargs):
        pass    # LocalStorage has no params for now...
        #self.index = ""
        #self.run = ""
        self.index_dir = None
        self.run_dir = None
        self.index_made = False
        self.run_made = False
        

    def save(self, index: str = None, run: str = None, file_path: os.PathLike = None):
        if index is None:
            index = self.index
        if run is None:
            run = self.run

        if not self.index_made:
            self.makeIndexDir()
        if not self.run_made:
            self.makeRunDir()
        self.logger.info("[LocalStorage {storage}] Saving {filename} in index {index} with run {run}".format(
            storage=self.storage_url,
            filename = os.path.basename(file_path),
            index=index,
            run=run
        ))
        shutil.copy2(
            file_path,
            os.path.join(self.storage_url, index, run, os.path.basename(file_path))
        )
    
    def load(self, index: str = None, run: str = None, file_name: str = None, local_path: os.PathLike = None):
        if index is None:
            index = self.index
        if run is None:
            run = self.run
        self.logger.info("[LocalStorage {storage}] Loading {filename} from index {index} with run {run} into local file {target}".format(
            storage=self.storage_url,
            filename = file_name,
            target=os.path.basename(local_path),
            index=self.index,
            run=self.run
        ))
        
        shutil.copy2(
            os.path.join( self.storage_url, index, run, file_name), 
            local_path
        )

    def setIndex(self, index):
        self.index = index
        self.index_dir = os.path.join(self.storage_url, self.index)
        self.index_made = False
        
    
    def makeIndexDir(self):
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_made = True

    def getMostRecentRun(self):
        run_list = glob(os.path.join(self.index_dir, "run*"))
        try:
            run_max = max([int(item[3:]) for item in run_list])
        except (TypeError, ValueError):
            run_max = "-1"
        return run_max
    
    def getNextRun(self):
        return str(int(self.getMostRecentRun())+1)

    def setRun(self, run):
        if int(run) == -1:
            run = 0
        self.run = "run%s"%str(run)
        self.run_dir = os.path.join(self.storage_url, self.index, self.run)
        self.run_made = False
    
    def makeRunDir(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_made = True

    def exists(self, index: str, run: str, file_path: os.PathLike):
        return os.path.exists(os.path.join( self.storage_url, index, run,  os.path.basename(file_path)))
