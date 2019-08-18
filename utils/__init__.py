from . import web as web


def generate_save_names(cfg):
    MODEL_SAVE_NAME = "%s-v%i"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"))
    MODEL_SAVE_FOLDER = "%s-v%i-%s-%s"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
    LOGGER_SAVE_NAME = "%s-v%i-%s-%s-logger.log"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
    if cfg.get("SAVE.DRIVE_BACKUP"):
        CHECKPOINT_DIRECTORY = "./drive/My Drive/Vehicles/Models/" + MODEL_SAVE_FOLDER
    else:
        CHECKPOINT_DIRECTORY = ''
    return MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY

def fix_generator_arguments(cfg):
    if type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is int or type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is float:
        NORMALIZATION_MEAN = [cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")]*cfg.get("TRANSFORMATION.CHANNELS")
    if type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is int or type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is float:
        NORMALIZATION_STD = [cfg.get("TRANSFORMATION.NORMALIZATION_STD")]*cfg.get("TRANSFORMATION.CHANNELS")
    if type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is int or type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is float:
        RANDOM_ERASE_VALUE = [cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")]*cfg.get("TRANSFORMATION.CHANNELS")
    return NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE