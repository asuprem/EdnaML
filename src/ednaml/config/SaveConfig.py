from ednaml.config import BaseConfig
from ednaml.config.BackupOptionsConfig import BackupOptionsConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class SaveConfig(BaseConfig):
    MODEL_VERSION: int
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    BACKUP: BackupOptionsConfig
    DRIVE_BACKUP: BackupOptionsConfig
    LOG_BACKUP: BackupOptionsConfig
    MODEL_BACKUP: BackupOptionsConfig
    ARTIFACTS_BACKUP: BackupOptionsConfig
    METRICS_BACKUP: BackupOptionsConfig
    CONFIG_BACKUP: BackupOptionsConfig
    SAVE_FREQUENCY: int

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

        self.BACKUP = BackupOptionsConfig(save_dict.get("BACKUP", {}), defaults)
        #self.DRIVE_BACKUP = BackupOptionsConfig(save_dict.get("BACKUP", {}, defaults)
        self.LOG_BACKUP = BackupOptionsConfig(save_dict.get("LOG_BACKUP", save_dict.get("BACKUP", {})), defaults)
        self.MODEL_BACKUP = BackupOptionsConfig(save_dict.get("MODEL_BACKUP", save_dict.get("BACKUP", {})), defaults)
        self.ARTIFACTS_BACKUP = BackupOptionsConfig(save_dict.get("ARTIFACTS_BACKUP", save_dict.get("MODEL_BACKUP", save_dict.get("BACKUP", {}))), defaults)
        self.CONFIG_BACKUP = BackupOptionsConfig(save_dict.get("CONFIG_BACKUP", save_dict.get("BACKUP", {})), defaults)
        self.PLUGIN_BACKUP = BackupOptionsConfig(save_dict.get("PLUGIN_BACKUP", save_dict.get("MODEL_BACKUP", save_dict.get("BACKUP", {}))), defaults)
        self.METRICS_BACKUP = BackupOptionsConfig(save_dict.get("METRICS_BACKUP", save_dict.get("BACKUP", {})), defaults)

        #self.DRIVE_BACKUP = save_dict.get("DRIVE_BACKUP", defaults.DRIVE_BACKUP)
        #self.LOG_BACKUP = save_dict.get("LOG_BACKUP", defaults.LOG_BACKUP)




        self.SAVE_FREQUENCY = save_dict.get(
            "SAVE_FREQUENCY", defaults.SAVE_FREQUENCY
        )
        self.STEP_SAVE_FREQUENCY = save_dict.get(
            "STEP_SAVE_FREQUENCY", defaults.STEP_SAVE_FREQUENCY
        )
