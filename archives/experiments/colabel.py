# For running the colabel model.
# data input is <img, annotation1, annotation2, annotationk, label>
# Model is built with k branches
# Features emrged later...

import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary

"""
For CoLabel master model...
"""

@click.command()
@click.argument('config')
@click.option('--mode', default="train", help="Execution mode: [train/test]")
@click.option('--weights', default="", help="Path to weights if mode is test")
def main(config, mode, weights):
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)

    # NOTE: set up the base config file...
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    logger.info("*"*40);logger.info("");logger.info("")
    logger.info("Using the following configuration:")
    logger.info(config.export("yaml", indent=4))
    logger.info("");logger.info("");logger.info("*"*40)


    # TODO fix this for the random erase value...
    NORMALIZATION_MEAN, NORMALIZATION_STD, _ = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


    """ MODEL PARAMS """
    # This will setup the model weights and load the appropriate one given our configuration
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


    # ------------------ LOAD SAVED LOGGER IF EXISTS ----------------------------
    DRIVE_BACKUP = config.get("SAVE.DRIVE_BACKUP")
    if DRIVE_BACKUP:
        backup_logger = os.path.join(CHECKPOINT_DIRECTORY, LOGGER_SAVE_NAME)
        if os.path.exists(backup_logger):
            logger.info("Existing log file exists at %s. Will attempt to copy to local directory %s."%(backup_logger, MODEL_SAVE_FOLDER))
            shutil.copy2(backup_logger, MODEL_SAVE_FOLDER)
    else:
        # Check if there is a backup logger locally
        backup_logger = os.path.join(MODEL_SAVE_FOLDER,LOGGER_SAVE_NAME)
        if os.path.exists(backup_logger):
            logger.info("Existing log file exists at %s. Will attempt to append there."%backup_logger)

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs"%NUM_GPUS)



    # --------------------- BUILD GENERATORS ------------------------
    data_crawler_ = config.get("EXECUTION.CRAWLER", "CoLabelIntegratedDatasetCrawler")
    data_crawler = __import__("crawlers."+data_crawler_, fromlist=[data_crawler_])
    data_crawler = getattr(data_crawler, data_crawler_) # contains list of imgs inside in crawler.metadata["train"]["crawl"] -->[(img-path, img-class-id), (img-path, img-class-id), ...]

    from generators import CoLabelIntegratedDatasetGenerator
    logger.info("Crawling data folder %s"%config.get("DATASET.ROOT_DATA_FOLDER"))
    crawler = data_crawler(data_folder = config.get("DATASET.ROOT_DATA_FOLDER"), train_folder=config.get("DATASET.TRAIN_FOLDER"), test_folder = config.get("DATASET.TEST_FOLDER"), **{"logger":logger})
    train_generator = CoLabelIntegratedDatasetGenerator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                **TRAINDATA_KWARGS)
    train_generator.setup(crawler, mode='train',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), workers = config.get("TRANSFORMATION.WORKERS"))
    logger.info("Generated training data generator")
    TRAIN_CLASSES = config.get("MODEL.SOFTMAX_DIM", train_generator.num_entities)
    test_generator=  CoLabelIntegratedDatasetGenerator(    gpus=NUM_GPUS, 
                                            i_shape=config.get("DATASET.SHAPE"),
                                            normalization_mean=NORMALIZATION_MEAN, 
                                            normalization_std = NORMALIZATION_STD, 
                                            normalization_scale = 1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"),
                                            h_flip = 0, 
                                            t_crop = False, 
                                            rea = False)
    test_generator.setup(   crawler, 
                            mode='test', 
                            batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), 
                            workers=config.get("TRANSFORMATION.WORKERS"))
    NUM_CLASSES = train_generator.num_entities
    logger.info("Generated validation data/query generator")
