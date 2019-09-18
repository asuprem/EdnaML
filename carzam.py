# python carzam.py config/carzam_cars196.yml

"""

Multi-resolution object recognition by proxy

    - General, tradition approach >> given a dataset, train a classifier to recognize classes

    - How does drift occur?
        - new classes
        - new data
        - domain shift
            - new types of pictures
            - multi-resolution pictures
            - pictures from different types of cameras, e.g., that create new sub-pixel artifacts
            - adversarial images (similar to above)
        - concept drift
            - The logo itself changes (can be a slight change, or a radical change...)
            - new type of vehicle

    - Solutions for each problem mentioned
        - domain shift
            - new types of pictures:
                - train existing models with new images + old images
                - StyleGAN for converting new images to old type
            - multi-resolution/multi-scale images
                - look at some examples and stuff...
            - different camera types
                - Style-GAN for conversion
            - StyleGAN
                - for each styleGAN, use gaussian blurring (?) to 
                  limit adversarial exploitation
        - concept drift
            - logo itself changes OR new logo
                - learning by proxy
                - learn a proxy for collection of similar logoes
                - no need to update proxy; just

    
Proxy based learning


# This is for multi-resolution vehicle identification
# More generally, multi-resolution object recognition by proxy

"""

# Basic steps for BoxCar
# Apply it as a triplet problem...
# First test with Cars 196
# Then with Veri-Wild

# Applying EBKA -- future steps


# Using a triplet model -> distance matching
# With proxy

import os, shutil, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary, torchvision

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

    # --------------------- BUILD GENERATORS ------------------------
    from generators import SequencedGenerator
    
    pdb.set_trace()
    
    data_crawler_ = config.get("EXECUTION.CRAWLER")
    data_crawler = __import__("crawlers."+data_crawler_, fromlist=[data_crawler_])
    data_crawler = getattr(data_crawler, data_crawler_)
    
    crawler = data_crawler(data_folder = config.get("DATASET.ROOT_DATA_FOLDER"), train_folder=config.get("DATASET.TRAIN_FOLDER"), test_folder = config.get("DATASET.TEST_FOLDER"), query_folder=config.get("DATASET.QUERY_FOLDER"), **{"logger":logger})

    train_generator = SequencedGenerator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                **TRAINDATA_KWARGS)
    train_generator.setup(crawler, mode='train',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), instance = config.get("TRANSFORMATION.INSTANCES"), workers = config.get("TRANSFORMATION.WORKERS"))

    logger.info("Generated training data generator")
    TRAIN_CLASSES = train_generator.num_entities
    test_generator=  SequencedGenerator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                            normalization_mean=NORMALIZATION_MEAN, normalization_std = NORMALIZATION_STD, normalization_scale = 1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                            h_flip = 0, t_crop = False, rea = False)
    test_generator.setup(crawler, mode='test', batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), instance=config.get("TRANSFORMATION.INSTANCES"), workers=config.get("TRANSFORMATION.WORKERS"))
    QUERY_CLASSES = test_generator.num_entities
    logger.info("Generated validation data/query generator")

    # --------------------- INSTANTIATE MODEL ------------------------

    pdb.set_trace()















if __name__ == "__main__":
    main()