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

    # --------------------- BUILD GENERATORS ------------------------    
    
    data_generator_ = config.get("EXECUTION.GENERATOR")
    data_generator = __import__("generators."+data_generator_, fromlist=[data_generator_])
    data_generator = getattr(data_generator, data_generator_)
    

    data_crawler_ = config.get("EXECUTION.CRAWLER")
    data_crawler = __import__("crawlers."+data_crawler_, fromlist=[data_crawler_])
    data_crawler = getattr(data_crawler, data_crawler_)
    
    crawler = data_crawler(data_folder = config.get("DATASET.ROOT_DATA_FOLDER"), train_folder=config.get("DATASET.TRAIN_FOLDER"), test_folder = config.get("DATASET.TEST_FOLDER"), query_folder=config.get("DATASET.QUERY_FOLDER"), **{"logger":logger})

    train_generator = data_generator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                **TRAINDATA_KWARGS)
    train_generator.setup(crawler, mode='train',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), instance = config.get("TRANSFORMATION.INSTANCES"), workers = config.get("TRANSFORMATION.WORKERS"))

    logger.info("Generated training data generator")
    TRAIN_CLASSES = train_generator.num_entities
    test_generator=  data_generator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                            normalization_mean=NORMALIZATION_MEAN, normalization_std = NORMALIZATION_STD, normalization_scale = 1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                            h_flip = 0, t_crop = False, rea = False)
    test_generator.setup(crawler, mode='test', batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), instance=config.get("TRANSFORMATION.INSTANCES"), workers=config.get("TRANSFORMATION.WORKERS"))
    TEST_CLASSES = test_generator.num_entities
    logger.info("Generated validation data/query generator")

    # --------------------- INSTANTIATE MODEL ------------------------
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("EXECUTION.MODEL_BUILDER"))
    logger.info("Loaded {} from {} to build CarZam model".format(config.get("EXECUTION.MODEL_BUILDER"), "models"))

    carzam_model = model_builder(   arch=config.get("MODEL.MODEL_ARCH"), \
                                    base=config.get("MODEL.MODEL_BASE"), \
                                    weights=MODEL_WEIGHTS, \
                                    embedding_dimensions = config.get("MODEL.EMBEDDING_DIMENSIONS"), \
                                    normalization = config.get("MODEL.MODEL_NORMALIZATION"), \
                                    **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model with {} architecture".format(config.get("MODEL.MODEL_ARCH")))

    if mode == "test":
        carzam_model.load_state_dict(torch.load(weights))
        carzam_model.cuda()
        carzam_model.eval()
    else:
        if weights != "":   # Load weights if train and starting from a another model base...
            logger.info("Commencing partial model load from {}".format(weights))
            carzam_model.partial_load(weights)
            logger.info("Completed partial model load from {}".format(weights))
        carzam_model.cuda()
        logger.info(torchsummary.summary(carzam_model, input_size=(3, *config.get("DATASET.SHAPE"))))

    # --------------------- INSTANTIATE LOSS ------------------------
    from loss import CarZamLossBuilder as LossBuilder
    loss_function = LossBuilder(loss_functions=config.get("LOSS.LOSSES"), loss_lambda=config.get("LOSS.LOSS_LAMBDAS"), loss_kwargs=config.get("LOSS.LOSS_KWARGS"), **{"logger":logger})
    logger.info("Built loss function")

    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    optimizer_builder = __import__("optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER"))
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER"))
    logger.info("Loaded {} from {} to build Optimizer model".format(config.get("EXECUTION.OPTIMIZER_BUILDER"), "optimizer"))

    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"), lr_bias = config.get("OPTIMIZER.LR_BIAS_FACTOR"), weight_decay=config.get("OPTIMIZER.WEIGHT_DECAY"), weight_bias=config.get("OPTIMIZER.WEIGHT_BIAS_FACTOR"), gpus=NUM_GPUS)
    optimizer = OPT.build(carzam_model, config.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Build optimizer")

    # --------------------- INSTANTIATE SCHEDULER ------------------------
    try:
        scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
        scheduler = getattr(scheduler, config.get("SCHEDULER.LR_SCHEDULER"))
    except (ModuleNotFoundError, AttributeError):
        scheduler_ = config.get("SCHEDULER.LR_SCHEDULER")
        scheduler = __import__("scheduler."+scheduler_, fromlist=[scheduler_])
        scheduler = getattr(scheduler, scheduler_)
    scheduler = scheduler(optimizer, last_epoch = -1, **json.loads(config.get("SCHEDULER.LR_KWARGS")))
    logger.info("Built scheduler")

    # --------------------- DRIVE BACKUP ------------------------
    if DRIVE_BACKUP:
        fl_list = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, "*.pth"))
    else:
        fl_list = glob.glob(os.path.join(MODEL_SAVE_FOLDER, "*.pth"))
    _re = re.compile(r'.*epoch([0-9]+)\.pth')
    previous_stop = [int(item[1]) for item in [_re.search(item) for item in fl_list] if item is not None]
    if len(previous_stop) == 0:
        previous_stop = 0
    else:
        previous_stop = max(previous_stop) + 1
        logger.info("Previous stop detected. Will attempt to resume from epoch %i"%previous_stop)

    # --------------------- PERFORM TRAINING ------------------------
    trainer = __import__("trainer", fromlist=["*"])
    trainer = getattr(model_builder, config.get("EXECUTION.TRAINER"))
    
    loss_stepper = trainer(model=carzam_model, loss_fn = loss_function, optimizer = optimizer, scheduler = scheduler, train_loader = train_generator.dataloader, test_loader = test_generator.dataloader, queries = TEST_CLASSES, epochs = config.get("EXECUTION.EPOCHS"), logger = logger)
    loss_stepper.setup(step_verbose = config.get("LOGGING.STEP_VERBOSE"), save_frequency=config.get("SAVE.SAVE_FREQUENCY"), test_frequency = config.get("EXECUTION.TEST_FREQUENCY"), save_directory = MODEL_SAVE_FOLDER, save_backup = DRIVE_BACKUP, backup_directory = CHECKPOINT_DIRECTORY, gpus=NUM_GPUS,fp16 = config.get("OPTIMIZER.FP16"), model_save_name = MODEL_SAVE_NAME, logger_file = LOGGER_SAVE_NAME)
    if mode == 'train':
      loss_stepper.train(continue_epoch=previous_stop)
    elif mode == 'test':
      loss_stepper.evaluate()
    else:
      raise NotImplementedError()

    # TODO update evaluate function (NMI) and loss checker

    """Notes

    proxy is part of model
    during training, we return both embeddings + proxies
    During loss calculation, we calculate distance between embeddings and proxies...
    Then loss.backward should update both proxies and model itself...
    (Or should proxies be independent??? -- no, model keeps its proxies)


    During evaluation, we can proceed normally...for now, work on  integrating proxies into the model AND into batch_kwargs TODO


    """







if __name__ == "__main__":
    main()