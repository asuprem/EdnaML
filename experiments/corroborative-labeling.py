import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary


"""
Cor Co-Labeling, we need all configs of the component ensembles. Easiest to just load the configs directly, no?
"""

@click.command()
@click.argument('config')
@click.option('--ensemble', multiple=True, help="List of ensemble model components configs")
@click.option('--weights', multiple=True, help="Path to weights, for each member of ensemble")
def main(config, ensemble, weights):
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)

    # NOTE: set up the base config file...
    # potentially don't need these???? because we are performing a deployment, not a model training...
    # But we will need the Logger....
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    logger.info("*"*40);logger.info("");logger.info("")
    logger.info("Using the following configuration:")
    logger.info(config.export("yaml", indent=4))
    logger.info("");logger.info("");logger.info("*"*40)

    # TODO fix this for the random erase value...
    NORMALIZATION_MEAN, NORMALIZATION_STD, _ = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs"%NUM_GPUS)

    #------- DATACRAWLER ---------------------------------------------------------
    data_crawler_ = config.get("EXECUTION.CRAWLER", "CoLabelVehicleColorCrawler")
    data_crawler = __import__("crawlers."+data_crawler_, fromlist=[data_crawler_])
    data_crawler = getattr(data_crawler, data_crawler_) # contains list of imgs inside in crawler.metadata["train"]["crawl"] -->[(img-path, img-class-id), (img-path, img-class-id), ...]

    from generators import CoLabelGenerator
    logger.info("Crawling data folder %s"%config.get("DATASET.ROOT_DATA_FOLDER"))
    crawler = data_crawler(data_folder = config.get("DATASET.ROOT_DATA_FOLDER"), train_folder=config.get("DATASET.TRAIN_FOLDER"), test_folder = config.get("DATASET.TEST_FOLDER"), **{"logger":logger})
    test_generator=  CoLabelGenerator(    gpus=NUM_GPUS, 
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

    logger.info("Generated validation data generator")

    from deployments import CoLabelEnsemble
    ensemble = CoLabelEnsemble(logger=logger)
    ensemble.addModels(config, ensemble)


    # We need a data loader, then predict the data with this...

    #ensemble.predict(test_generator.dataloader)

    return ensemble, crawler.classes


