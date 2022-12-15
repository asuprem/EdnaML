from typing import List
from glob import glob
import os, shutil
import re, warnings
from ednaml.storage.BaseStorage import BaseStorage
from ednaml.utils import ERSKey, KeyMethods, StorageArtifactType


class MongoStorage(BaseStorage):
    def apply(self, storage_url: str, **kwargs):
        return super().apply(storage_url, **kwargs)