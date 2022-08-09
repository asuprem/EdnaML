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
        shutil.copy2(
            file_path,
            os.path.join(self.storage_url, index, run, os.path.basename(file_path))
        )
    
    def load(self, index: str = None, run: str = None, file_name: str = None, local_path: os.PathLike = None):
        if index is None:
            index = self.index
        if run is None:
            run = self.run
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
            run_max = "0"
        return run_max

    def setRun(self, run):
        self.run = str(run)
        self.run_dir = os.path.join(self.index_dir, "run%s"%self.run)
        self.run_made = False
    
    def makeRunDir(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_made = True

    def exists(self, index: str, run: str, file_path: os.PathLike):
        return os.path.exists(os.path.join( self.storage_url, index, run,  os.path.basename(file_path)))
