from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class BackupOptionsConfig(BaseConfig):
    BACKUP: bool
    STORAGE_NAME: str
    FREQUENCY: int
    
    def __init__(self, save_dict, defaults: ConfigDefaults):
        self.BACKUP = save_dict.get(
            "BACKUP", defaults.BACKUP_PERFORM
        )
        self.STORAGE_NAME = save_dict.get(
            "STORAGE_NAME", defaults.BACKUP_LOCATION
        )
        self.FREQUENCY = save_dict.get(
            "FREQUENCY", defaults.BACKUP_FREQUENCY
        )
        self.FREQUENCY_STEP = save_dict.get(
            "FREQUENCY_STEP", defaults.BACKUP_FREQUENCY_STEP
        )