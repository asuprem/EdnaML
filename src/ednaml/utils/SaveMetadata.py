import os
from typing import Tuple
from ednaml.config.EdnaMLConfig import EdnaMLConfig


class SaveMetadata:
    """Contains paths for saving EdnaML models, loggers, metadata, and training information"""

    MODEL_VERSION: str
    MODEL_CORE_NAME: str
    MODEL_BACKBONE: str
    MODEL_QUALIFIER: str

    # TODO remove dependenency on these...
    MODEL_SAVE_NAME: str
    MODEL_SAVE_FOLDER: str
    LOGGER_SAVE_NAME: str

    def __init__(self, cfg: EdnaMLConfig, **kwargs):
        (
            self.MODEL_SAVE_NAME,
            self.MODEL_SAVE_FOLDER,
            self.LOGGER_SAVE_NAME,
        ) = SaveMetadata.generate_save_names_from_config(cfg, **kwargs)

        self.save_ref = cfg.SAVE

    @property
    def MODEL_VERSION(self):
        return self.save_ref.MODEL_VERSION

    @property
    def MODEL_CORE_NAME(self):
        return self.save_ref.MODEL_CORE_NAME

    @property
    def MODEL_BACKBONE(self):
        return self.save_ref.MODEL_BACKBONE

    @property
    def MODEL_QUALIFIER(self):
        return self.save_ref.MODEL_QUALIFIER

    @staticmethod  # We will need to not use this anymore...
    def generate_save_names_from_config(cfg: EdnaMLConfig, **kwargs) -> Tuple[str]:

        MODEL_SAVE_NAME = "%s-v%i" % (
            cfg.SAVE.MODEL_CORE_NAME,
            cfg.SAVE.MODEL_VERSION,
        )
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

        return (
            MODEL_SAVE_NAME,
            MODEL_SAVE_FOLDER,
            LOGGER_SAVE_NAME,
        )
