


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


    def __init__(self, cfg):
        self.MODEL_SAVE_NAME, self.MODEL_SAVE_FOLDER, self.LOGGER_SAVE_NAME, self.CHECKPOINT_DIRECTORY = SaveMetadata.generate_save_names_from_config(cfg)

        self.MODEL_VERSION = cfg.get("SAVE.MODEL_VERSION")
        self.MODEL_CORE_NAME = cfg.get("SAVE.MODEL_CORE_NAME")
        self.MODEL_BACKBONE = cfg.get("SAVE.MODEL_BACKBONE")
        self.MODEL_QUALIFIER = cfg.get("SAVE.MODEL_QUALIFIER")
        self.DRIVE_BACKUP = cfg.get("SAVE.DRIVE_BACKUP")


    @staticmethod
    def generate_save_names_from_config(cfg):
        MODEL_SAVE_NAME = "%s-v%i"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"))
        MODEL_SAVE_FOLDER = "%s-v%i-%s-%s"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
        LOGGER_SAVE_NAME = "%s-v%i-%s-%s-logger.log"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
        if cfg.get("SAVE.DRIVE_BACKUP"):
            CHECKPOINT_DIRECTORY = cfg.get("SAVE.CHECKPOINT_DIRECTORY","./drive/My Drive/Vehicles/Models/") + MODEL_SAVE_FOLDER
        else:
            CHECKPOINT_DIRECTORY = ''
        return MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY