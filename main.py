# For VeRi
import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary

@click.command()
@click.argument('config')
@click.option('--mode', default="train", help="Execution mode: [train/test]")
@click.option('--weights', default="", help="Path to weights if mode is test")
def main(config, mode, weights):
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)
    
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    logger = logging.getLogger(MODEL_SAVE_FOLDER)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOGGER_SAVE_NAME)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(msecs)d %(message)s',datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    cs.setFormatter(logging.Formatter('%(asctime)s-%(msecs)d %(message)s',datefmt="%H:%M:%S"))
    logger.addHandler(cs)

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


    # Load previousely saved logger, if it exists:
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

    data_crawler_ = config.get("EXECUTION.CRAWLER", "VeRiDataCrawler")
    data_crawler = __import__("crawlers."+data_crawler_, fromlist=[data_crawler_])
    data_crawler = getattr(data_crawler, data_crawler_)

    from generators import SequencedGenerator
    logger.info("Crawling data folder %s"%config.get("DATASET.ROOT_DATA_FOLDER"))
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
    model_builder_ = data_crawler_ = config.get("EXECUTION.MODEL_BUILDER", "veri_model_builder")
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, model_builder_)
    logger.info("Loaded {} from {} to build ReID model".format(model_builder_, "models"))

    reid_model = model_builder(    arch = config.get("MODEL.MODEL_ARCH"), \
                                        base=config.get("MODEL.MODEL_BASE"), \
                                        weights=MODEL_WEIGHTS, \
                                        soft_dimensions = TRAIN_CLASSES, \
                                        embedding_dimensions = config.get("MODEL.EMB_DIM"), \
                                        normalization = config.get("MODEL.MODEL_NORMALIZATION"), \
                                        **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")

    if mode == "test":
        reid_model.load_state_dict(torch.load(weights))
        reid_model.cuda()
        reid_model.eval()
    else:
        if weights != "":   # Load weights if train and starting from a another model base...
            logger.info("Commencing partial model load from {}".format(weights))
            reid_model.partial_load(weights)
            logger.info("Completed partial model load from {}".format(weights))
        reid_model.cuda()
        logger.info(torchsummary.summary(reid_model, input_size=(3, *config.get("DATASET.SHAPE"))))
    # --------------------- INSTANTIATE LOSS ------------------------
    from loss import ReIDLossBuilder
    loss_function = ReIDLossBuilder(loss_functions=config.get("LOSS.LOSSES"), loss_lambda=config.get("LOSS.LOSS_LAMBDAS"), loss_kwargs=config.get("LOSS.LOSS_KWARGS"), **{"logger":logger})
    logger.info("Built loss function")

    # --------------------- INSTANTIATE LOSS OPTIMIZER --------------
    from optimizer.StandardLossOptimizer import StandardLossOptimizer as loss_optimizer

    LOSS_OPT = loss_optimizer(  base_lr=config.get("LOSS_OPTIMIZER.BASE_LR", config.get("OPTIMIZER.BASE_LR")), 
                                lr_bias = config.get("LOSS_OPTIMIZER.LR_BIAS_FACTOR", config.get("OPTIMIZER.LR_BIAS_FACTOR")), 
                                weight_decay= config.get("LOSS_OPTIMIZER.WEIGHT_DECAY", config.get("OPTIMIZER.WEIGHT_DECAY")), 
                                weight_bias= config.get("LOSS_OPTIMIZER.WEIGHT_BIAS_FACTOR", config.get("OPTIMIZER.WEIGHT_BIAS_FACTOR")), 
                                gpus=NUM_GPUS)
    loss_optimizer = LOSS_OPT.build(loss_builder=loss_function,
                                    name=config.get("LOSS_OPTIMIZER.OPTIMIZER_NAME", config.get("OPTIMIZER.OPTIMIZER_NAME")),
                                    **json.loads(config.get("LOSS_OPTIMIZER.OPTIMIZER_KWARGS", config.get("OPTIMIZER.OPTIMIZER_KWARGS"))))
    logger.info("Built loss optimizer")
    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    optimizer_builder_ = config.get("EXECUTION.OPTIMIZER_BUILDER", "OptimizerBuilder")
    optimizer_builder = __import__("optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, optimizer_builder_)
    logger.info("Loaded {} from {} to build Optimizer model".format(optimizer_builder_, "optimizer"))

    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"), lr_bias = config.get("OPTIMIZER.LR_BIAS_FACTOR"), weight_decay=config.get("OPTIMIZER.WEIGHT_DECAY"), weight_bias=config.get("OPTIMIZER.WEIGHT_BIAS_FACTOR"), gpus=NUM_GPUS)
    optimizer = OPT.build(reid_model, config.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Built optimizer")
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

    # ------------------- INSTANTIATE LOSS SCHEEDULER ---------------------
    loss_scheduler = None
    if loss_optimizer is not None:  # In case loss has no differentiable paramters
        try:
            loss_scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
            loss_scheduler = getattr(loss_scheduler, config.get("LOSS_SCHEDULER.LR_SCHEDULER", config.get("SCHEDULER.LR_SCHEDULER")))
        except (ModuleNotFoundError, AttributeError):
            loss_scheduler_ = config.get("LOSS_SCHEDULER.LR_SCHEDULER", config.get("SCHEDULER.LR_SCHEDULER"))
            loss_scheduler = __import__("scheduler."+loss_scheduler_, fromlist=[loss_scheduler_])
            loss_scheduler = getattr(loss_scheduler, loss_scheduler_)
        loss_scheduler = loss_scheduler(loss_optimizer, last_epoch = -1, **json.loads(config.get("LOSS_SCHEDULER.LR_KWARGS", config.get("SCHEDULER.LR_KWARGS"))))
        logger.info("Built loss scheduler")
    else:
        loss_scheduler = None
    
    # ---------------------------- SETUP BACKUP PATH -------------------------
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
    trainer_ = config.get("EXECUTION.TRAINER","SimpleTrainer")
    trainer = __import__("trainer", fromlist=["*"])
    trainer = getattr(trainer, trainer_)
    logger.info("Loaded {} from {} to build Trainer".format(trainer_, "trainer"))

    loss_stepper = trainer(model=reid_model, loss_fn = loss_function, optimizer = optimizer, loss_optimizer = loss_optimizer, scheduler = scheduler, loss_scheduler = loss_scheduler, train_loader = train_generator.dataloader, test_loader = test_generator.dataloader, queries = QUERY_CLASSES, epochs = config.get("EXECUTION.EPOCHS"), logger = logger)
    loss_stepper.setup(step_verbose = config.get("LOGGING.STEP_VERBOSE"), save_frequency=config.get("SAVE.SAVE_FREQUENCY"), test_frequency = config.get("EXECUTION.TEST_FREQUENCY"), save_directory = MODEL_SAVE_FOLDER, save_backup = DRIVE_BACKUP, backup_directory = CHECKPOINT_DIRECTORY, gpus=NUM_GPUS,fp16 = config.get("OPTIMIZER.FP16"), model_save_name = MODEL_SAVE_NAME, logger_file = LOGGER_SAVE_NAME)
    if mode == 'train':
      loss_stepper.train(continue_epoch=previous_stop)
    elif mode == 'test':
      loss_stepper.evaluate()
    else:
      raise NotImplementedError()
    

if __name__ == "__main__":
    main()