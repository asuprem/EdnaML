from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class BackupConfig(BaseConfig):
    STORAGE: str
    FREQUENCY: int
    def __init__(self, config_dict, defaults: ConfigDefaults):
        self.STORAGE = config_dict.get("STORAGE", "no_storage_provided")
        self.FREQUENCY = config_dict.get("FREQUENCY", -1)   # -1: never. 0: Once. >0: At save frequency, e.g. whenever save is triggered. We can deal with other variations later.

class SaveConfig(BaseConfig):
    MODEL_VERSION: int
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    DRIVE_BACKUP: bool
    SAVE_FREQUENCY: int

    CONFIG_BACKUP: BackupConfig
    LOG_BACKUP: BackupConfig
    MODEL_BACKUP: BackupConfig
    MODEL_ARTIFACTS_BACKUP: BackupConfig
    MODEL_PLUGIN_BACKUP: BackupConfig
    METRICS_BACKUP: BackupConfig

    
    def __init__(self, save_dict, defaults: ConfigDefaults):
        self.MODEL_VERSION = save_dict.get(
            "MODEL_VERSION", defaults.MODEL_VERSION
        )
        self.MODEL_CORE_NAME = save_dict.get(
            "MODEL_CORE_NAME", defaults.MODEL_CORE_NAME
        )
        self.MODEL_BACKBONE = save_dict.get(
            "MODEL_BACKBONE", defaults.MODEL_BACKBONE
        )
        self.MODEL_QUALIFIER = save_dict.get(
            "MODEL_QUALIFIER", defaults.MODEL_QUALIFIER
        )
        self.DRIVE_BACKUP = save_dict.get("DRIVE_BACKUP", defaults.DRIVE_BACKUP)
        self.SAVE_FREQUENCY = save_dict.get(
            "SAVE_FREQUENCY", defaults.SAVE_FREQUENCY
        )


        self.CONFIG_BACKUP = BackupConfig(save_dict.get("CONFIG_BACKUP", {}), defaults=defaults)
        self.LOG_BACKUP = BackupConfig(save_dict.get("LOG_BACKUP", {}), defaults=defaults)
        self.MODEL_BACKUP = BackupConfig(save_dict.get("MODEL_BACKUP", {}), defaults=defaults)
        self.MODEL_ARTIFACTS_BACKUP = BackupConfig(save_dict.get("MODEL_ARTIFACTS_BACKUP", {}), defaults=defaults)
        self.MODEL_PLUGIN_BACKUP = BackupConfig(save_dict.get("MODEL_PLUGIN_BACKUP", {}), defaults=defaults)
        self.METRICS_BACKUP = BackupConfig(save_dict.get("METRICS_BACKUP", {}), defaults=defaults)
