import os
from typing import Tuple
from ednaml.config.EdnaMLConfig import EdnaMLConfig


class SaveMetadata:
    """Contains paths for saving EdnaML models, loggers, metadata, and training information
    """

    MODEL_VERSION: str
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str
    DRIVE_BACKUP: bool

    MODEL_SAVE_NAME: str
    MODEL_SAVE_FOLDER: str
    LOGGER_SAVE_NAME: str
    CHECKPOINT_DIRECTORY: str

    def __init__(self, cfg: EdnaMLConfig, **kwargs):
        (
            self.MODEL_SAVE_NAME,
            self.MODEL_SAVE_FOLDER,
            self.LOGGER_SAVE_NAME,
            self.CHECKPOINT_DIRECTORY,
        ) = SaveMetadata.generate_save_names_from_config(cfg, **kwargs)

        self.MODEL_VERSION = cfg.SAVE.MODEL_VERSION
        self.MODEL_CORE_NAME = cfg.SAVE.MODEL_CORE_NAME
        self.MODEL_BACKBONE = cfg.SAVE.MODEL_BACKBONE
        self.MODEL_QUALIFIER = cfg.SAVE.MODEL_QUALIFIER
        self.DRIVE_BACKUP = cfg.SAVE.DRIVE_BACKUP

    @staticmethod
    def generate_save_names_from_config(cfg: EdnaMLConfig, **kwargs) -> Tuple[str]:

        MODEL_SAVE_NAME = "%s-v%i" % (cfg.SAVE.MODEL_CORE_NAME, cfg.SAVE.MODEL_VERSION)
        MODEL_SAVE_FOLDER = "%s-v%i-%s-%s" % (
            cfg.SAVE.MODEL_CORE_NAME,
            cfg.SAVE.MODEL_VERSION,
            cfg.SAVE.MODEL_BACKBONE,
            cfg.SAVE.MODEL_QUALIFIER,
        )
        if kwargs.get("logger_save_name", None) is not None:
            LOGGER_SAVE_NAME = kwargs.get("logger_save_name")
        else:
            LOGGER_SAVE_NAME = "%s-v%i-%s-%s-logger.log" % (
                cfg.SAVE.MODEL_CORE_NAME,
                cfg.SAVE.MODEL_VERSION,
                cfg.SAVE.MODEL_BACKBONE,
                cfg.SAVE.MODEL_QUALIFIER,
            )
        if cfg.SAVE.DRIVE_BACKUP:
            CHECKPOINT_DIRECTORY = os.path.join(
                cfg.SAVE.CHECKPOINT_DIRECTORY, MODEL_SAVE_FOLDER
            )
        else:
            CHECKPOINT_DIRECTORY = ""
        return (
            MODEL_SAVE_NAME,
            MODEL_SAVE_FOLDER,
            LOGGER_SAVE_NAME,
            CHECKPOINT_DIRECTORY,
        )
