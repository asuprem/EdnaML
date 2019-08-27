import os, shutil, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary

from models import vaegan_model_builder

@click.command()
@click.argument('config')
@click.option('--mode', default="train", help="Execution mode: [train/test]")
@click.option('--weights', default=".", help="Path to weights if mode is test")
def main(config, mode, weights):
    # Generate configuration
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)

    # Generate logger
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    logger.info("*"*40);logger.info("");logger.info("")
    logger.info("Using the following configuration:")
    logger.info(config.export("yaml", indent=4))
    logger.info("");logger.info("");logger.info("*"*40)

    NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}

    """ MODEL PARAMS """
    from utils import model_weights

    MODEL_WEIGHTS = None
    if config.get("MODEL.MODEL_BASE") in model_weights:
        if mode == "train":
            if os.path.exists(model_weights[config.get("MODEL.MODEL_BASE")][1]):
                pass
            else:
                logger.info("Model weights file {} does not exist. Downloading.".format(model_weights[config.get("MODEL.MODEL_BASE")][1]))
                utils.web.download(model_weights[config.get("MODEL.MODEL_BASE")][1], model_weights[config.get("MODEL.MODEL_BASE")][0])
            MODEL_WEIGHTS = model_weights[config.get("MODEL.MODEL_BASE")][1]
    else:
        raise NotImplementedError("Model %s is not available. Please choose one of the following: %s"%(config.get("MODEL.MODEL_BASE"), str(model_weights.keys())))

    """ Load previousely saved logger, if it exists """
    DRIVE_BACKUP = config.get("SAVE.DRIVE_BACKUP")
    if DRIVE_BACKUP:
        backup_logger = os.path.join(CHECKPOINT_DIRECTORY, LOGGER_SAVE_NAME)
        if os.path.exists(backup_logger):
            shutil.copy2(backup_logger, ".")
    else:
        backup_logger = None

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs"%NUM_GPUS)

    """ DATA GENERATORS """
    """ TODO TODO TODO  """


    # --------------------- INSTANTIATE MODEL ------------------------
    reid_model = vaegan_model_builder(
                                **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")






if __name__ == "__main__":
    main()