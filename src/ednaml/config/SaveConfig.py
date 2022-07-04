from ednaml.config import BaseConfig
from ednaml.config.ConfigDefaults import ConfigDefaults


class SaveConfig(BaseConfig):
    MODEL_VERSION: int
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    DRIVE_BACKUP: bool
    LOG_BACKUP: bool
    SAVE_FREQUENCY: int
    CHECKPOINT_DIRECTORY: str

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
        self.LOG_BACKUP = save_dict.get("LOG_BACKUP", defaults.LOG_BACKUP)
        self.SAVE_FREQUENCY = save_dict.get(
            "SAVE_FREQUENCY", defaults.SAVE_FREQUENCY
        )
        self.CHECKPOINT_DIRECTORY = save_dict.get(
            "CHECKPOINT_DIRECTORY", defaults.CHECKPOINT_DIRECTORY
        )
