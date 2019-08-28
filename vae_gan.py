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

    """ SETUP IMPORTS """
    #from crawlers import ReidDataCrawler
    #from generators import SequencedGenerator
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("EXECUTION.MODEL_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.MODEL_BUILDER"), "models"))

    #from loss import LossBuilder
    
    optimizer_builder = __import__("optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.OPTIMIZER_BUILDER"), "optimizer"))

    #from trainer import SimpleTrainer


    NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


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
    vaegan_model = vaegan_model_builder(  latent_dimensions = config.get("MODEL.LATENT_DIMENSIONS"), \
                                        **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")

    if mode == "test":
        vaegan_model.load_state_dict(torch.load(weights))
        vaegan_model.cuda()
        vaegan_model.eval()
    else:
        vaegan_model.cuda()
        logger.info(torchsummary.summary(vaegan_model, input_size=(3, *config.get("DATASET.SHAPE"))))


    # --------------------- INSTANTIATE LOSS ------------------------
    # loss_function = LossBuilder(loss_functions=config.get("LOSS.LOSSES"), loss_lambda=config.get("LOSS.LOSS_LAMBDAS"), loss_kwargs=config.get("LOSS.LOSS_KWARGS"), **{"logger":logger})
    # logger.info("Built loss function")
    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"))
    optimizer = OPT.build(vaegan_model, config.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Built optimizer")






if __name__ == "__main__":
    main()