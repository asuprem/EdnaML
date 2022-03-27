import os, shutil, logging, glob, re, pdb, json
import kaptan
import click
from datareaders import DataReader
import utils
import torch
from torchinfo import summary
from utils.LabelMetadata import LabelMetadata


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
    data_reader_class = config.get("EXECUTION.DATAREADER.DATAREADER", "VehicleColor")
    data_reader = __import__("datareaders."+data_reader_class, fromlist=[data_reader_class])
    data_reader: DataReader = getattr(data_reader, data_reader_class) # contains list of imgs inside in crawler.metadata["train"]["crawl"] -->[(img-path, img-class-id), (img-path, img-class-id), ...]

    # data_crawler is now data_reader.CRAWLER
    logger.info("Reading data with DataReader %s"%data_reader_class)
    crawler = data_reader.CRAWLER(logger=logger, **config.get("EXECUTION.DATAREADER.CRAWLER_ARGS"))
    

    # Update the generator...if needed
    new_generator_class = config.get("EXECUTION.DATAREADER.GENERATOR", None)
    if new_generator_class is not None:
        new_generator = __import__("generators."+new_generator_class, fromlist=[new_generator_class])
        data_reader.GENERATOR = getattr(new_generator, new_generator_class)
    
    train_generator = data_reader.GENERATOR(gpus=NUM_GPUS, i_shape=config.get("TRANSFORMATION.SHAPE"), \
                                normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                rea_value=config.get("TRANSFORMATION.RANDOM_ERASE_VALUE"), **config.get("EXECUTION.DATAREADER.GENERATOR_ARGS"))
    train_generator.setup(crawler, mode='train',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), workers = config.get("TRANSFORMATION.WORKERS"), **config.get("EXECUTION.DATAREADER.DATASET_ARGS"))
    logger.info("Generated training data generator")
    # TODO we need t fix the dataset part, where it is defined inside the generator file... we need to move it elsewhere into datasets/<>
    # TODO, second, we need to fix the TRAIN_CLASSES thing, and how to obtain it
    # TODO Third, we need to fix the embedding dim, softmax_dim, and any other dim debacle...
    labelMetadata:LabelMetadata = train_generator.num_entities
    print("Running classification model with classes:", labelMetadata)
    test_generator=  data_reader.GENERATOR( gpus=NUM_GPUS, 
                                            i_shape=config.get("TRANSFORMATION.SHAPE"),
                                            normalization_mean=NORMALIZATION_MEAN, 
                                            normalization_std = NORMALIZATION_STD, 
                                            normalization_scale = 1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"),
                                            h_flip = 0, 
                                            t_crop = False, 
                                            rea = False, **config.get("EXECUTION.DATAREADER.GENERATOR_ARGS"))
    test_generator.setup(   crawler, 
                            mode='test', 
                            batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), 
                            workers=config.get("TRANSFORMATION.WORKERS"), **config.get("EXECUTION.DATAREADER.DATASET_ARGS"))
    NUM_CLASSES = train_generator.num_entities
    logger.info("Generated validation data/query generator")


    # --------------------- INSTANTIATE MODEL ------------------------    
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("MODEL.BUILDER", "classification_model_builder"))
    logger.info("Loaded {} from {} to build model".format(config.get("MODEL.BUILDER", "classification_model_builder"), "models"))
    
    if type(config.get("MODEL.MODEL_KWARGS")) is dict:  # Compatibility with old configs. TODO fix all old configs.
        model_kwargs_dict = config.get("MODEL.MODEL_KWARGS")
    elif type(config.get("MODEL.MODEL_KWARGS")) is None:
        model_kwargs_dict = {}
    else:
        model_kwargs_dict = json.loads(config.get("MODEL.MODEL_KWARGS"))

    # TODO!!!!!!!
    model = model_builder(      arch = config.get("MODEL.MODEL_ARCH"), \
                                base=config.get("MODEL.MODEL_BASE"), \
                                weights=MODEL_WEIGHTS, \
                                metadata = labelMetadata, \
                                normalization = config.get("MODEL.MODEL_NORMALIZATION"), \
                                **model_kwargs_dict)
    logger.info("Finished instantiating model with {} architecture".format(config.get("MODEL.MODEL_ARCH")))

    if mode == "test":
        model.load_state_dict(torch.load(weights))
        model.cuda()
        model.eval()
    else:
        if weights != "":   # Load weights if train and starting from a another model base...
            logger.info("Commencing partial model load from {}".format(weights))
            model.partial_load(weights)
            logger.info("Completed partial model load from {}".format(weights))
        model.cuda()
        model_summary = summary(model, 
                      input_size=( config.get("TRANSFORMATION.BATCH_SIZE"), config.get("TRANSFORMATION.CHANNELS"), *config.get("TRANSFORMATION.SHAPE")),
                      col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                      depth=4,
                      mode= "train",
                      verbose= 2)
        logger.info(str(model_summary))


    # --------------------- INSTANTIATE LOSS ------------------------
    from loss import ClassificationLossBuilder
    # NOTE, we need to edit this to reflect the fact that LOSS is now an array.
    # We need to create a loss function for each lonn in array
    # Then pass in this loss array to the trainer function
    # In training, each loss function set is used for each output from the model...
    # NOTE the list of losses should correspond to the model outputs for a multi-branch (or any type of model)
    # This means whichever model is used, maybe we need to specify what the model outputs will be in the config as well, either through some templating method
    # or otherwise. Templating method would be that if we want to add a model from model_builder, then the model_builder function would have the template we need
    # to use in the config file for that model_builder to work properly. So a multi-branch model builder will tell us that the template should correspond to multiple
    # outputs...
    loss_function_array = [
        ClassificationLossBuilder(  loss_functions=loss_item["LOSSES"], 
                                    loss_lambda=loss_item["LAMBDAS"], 
                                    loss_kwargs=loss_item["KWARGS"], 
                                    name=loss_item.get("NAME", None),
                                    label=loss_item.get("LABEL", None),
                                    metadata=labelMetadata,
                                    **{"logger":logger})
        for loss_item in config.get("LOSS")
    ]
    logger.info("Built loss function")

    # --------------------- INSTANTIATE LOSS OPTIMIZER --------------
    # NOTE different optimizer for each loss...
    from optimizer.StandardLossOptimizer import StandardLossOptimizer as loss_optimizer

    LOSS_OPT = [loss_optimizer(  base_lr=config.get("LOSS_OPTIMIZER.BASE_LR", config.get("OPTIMIZER.BASE_LR")), 
                                lr_bias = config.get("LOSS_OPTIMIZER.LR_BIAS_FACTOR", config.get("OPTIMIZER.LR_BIAS_FACTOR")), 
                                weight_decay= config.get("LOSS_OPTIMIZER.WEIGHT_DECAY", config.get("OPTIMIZER.WEIGHT_DECAY")), 
                                weight_bias= config.get("LOSS_OPTIMIZER.WEIGHT_BIAS_FACTOR", config.get("OPTIMIZER.WEIGHT_BIAS_FACTOR")), 
                                gpus=NUM_GPUS) for _ in config.get("LOSS")]
    # Note: build returns None if there are no differentiable parameters
    loss_optimizer_array = [item.build(loss_builder=loss_function_array[idx],
                                    name=config.get("LOSS_OPTIMIZER.OPTIMIZER_NAME", config.get("OPTIMIZER.OPTIMIZER_NAME")),
                                    **json.loads(config.get("LOSS_OPTIMIZER.OPTIMIZER_KWARGS", config.get("OPTIMIZER.OPTIMIZER_KWARGS"))))
                                    for idx,item in enumerate(LOSS_OPT)]
    logger.info("Built loss optimizer")


    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    optimizer_builder = __import__("optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER", "BaseOptimizer"))
    logger.info("Loaded {} from {} to build Optimizer model".format(config.get("EXECUTION.OPTIMIZER_BUILDER", "OptimizerBuilder"), "optimizer"))

    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"), lr_bias = config.get("OPTIMIZER.LR_BIAS_FACTOR"), weight_decay=config.get("OPTIMIZER.WEIGHT_DECAY"), weight_bias=config.get("OPTIMIZER.WEIGHT_BIAS_FACTOR"), gpus=NUM_GPUS)
    optimizer = OPT.build(model, config.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Built optimizer")

    # --------------------- INSTANTIATE SCHEDULER ------------------------
    try:    # We first check if scheduler is part of torch's provided schedulers.
        scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
        scheduler = getattr(scheduler, config.get("SCHEDULER.LR_SCHEDULER"))
    except (ModuleNotFoundError, AttributeError):   # If it fails, then we try to import from schedulers implemented in scheduler/ folder
        scheduler_ = config.get("SCHEDULER.LR_SCHEDULER")
        scheduler = __import__("scheduler."+scheduler_, fromlist=[scheduler_])
        scheduler = getattr(scheduler, scheduler_)
    scheduler = scheduler(optimizer, last_epoch = -1, **json.loads(config.get("SCHEDULER.LR_KWARGS")))
    logger.info("Built scheduler")


    # ------------------- INSTANTIATE LOSS SCHEEDULER ---------------------
    loss_scheduler = [None]*len(loss_optimizer_array)
    for idx, _ in enumerate(loss_optimizer_array):
        if loss_optimizer_array[idx] is not None:  # In case loss has no differentiable paramters
            try:
                loss_scheduler[idx] = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
                loss_scheduler[idx] = getattr(loss_scheduler[idx], config.get("LOSS_SCHEDULER.LR_SCHEDULER", config.get("SCHEDULER.LR_SCHEDULER")))
            except (ModuleNotFoundError, AttributeError):
                loss_scheduler_ = config.get("LOSS_SCHEDULER.LR_SCHEDULER", config.get("SCHEDULER.LR_SCHEDULER"))
                loss_scheduler[idx] = __import__("scheduler."+loss_scheduler_, fromlist=[loss_scheduler_])
                loss_scheduler[idx] = getattr(loss_scheduler[idx], loss_scheduler_)
            loss_scheduler[idx] = loss_scheduler[idx](loss_optimizer_array[idx], last_epoch = -1, **json.loads(config.get("LOSS_SCHEDULER.LR_KWARGS", config.get("SCHEDULER.LR_KWARGS"))))
            logger.info("Built loss scheduler")
        else:
            loss_scheduler[idx] = None
    

    # ---------------------------- SETUP BACKUP PATH -------------------------
    if DRIVE_BACKUP:
        fl_list = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, "*.pth"))
    else:
        fl_list = glob.glob(os.path.join(MODEL_SAVE_FOLDER, "*.pth"))
    _re = re.compile(r'.*epoch([0-9]+)\.pth')
    previous_stop = [int(item[1]) for item in [_re.search(item) for item in fl_list] if item is not None]
    if len(previous_stop) == 0:
        previous_stop = 0
        logger.info("No previous stop detected. Will start from epoch 0")
    else:
        previous_stop = max(previous_stop) + 1
        logger.info("Previous stop detected. Will attempt to resume from epoch %i"%previous_stop)

    # --------------------- PERFORM TRAINING ------------------------
    ExecutionTrainer = __import__("trainer", fromlist=["*"])
    ExecutionTrainer = getattr(ExecutionTrainer, config.get("EXECUTION.TRAINER","ClassificationTrainer"))
    logger.info("Loaded {} from {} to build Trainer".format(config.get("EXECUTION.TRAINER","ClassificationTrainer"), "trainer"))

    trainer = ExecutionTrainer( model=model, 
                            loss_fn = loss_function_array, 
                            optimizer = optimizer, 
                            loss_optimizer = loss_optimizer_array, 
                            scheduler = scheduler, 
                            loss_scheduler = loss_scheduler, 
                            train_loader = train_generator.dataloader, 
                            test_loader = test_generator.dataloader, 
                            epochs = config.get("EXECUTION.EPOCHS"), 
                            skipeval = config.get("EXECUTION.SKIPEVAL"),
                            logger = logger, crawler=crawler,
                            config = config,
                            labels = labelMetadata)

    trainer.buildMetadata(crawler=crawler.classes, config=json.loads(config.export("json")))
    trainer.setup( step_verbose = config.get("LOGGING.STEP_VERBOSE"), 
                        save_frequency=config.get("SAVE.SAVE_FREQUENCY"), 
                        test_frequency = config.get("EXECUTION.TEST_FREQUENCY"), 
                        save_directory = MODEL_SAVE_FOLDER, 
                        save_backup = DRIVE_BACKUP, 
                        backup_directory = CHECKPOINT_DIRECTORY, 
                        gpus=NUM_GPUS,
                        fp16 = config.get("OPTIMIZER.FP16"), 
                        model_save_name = MODEL_SAVE_NAME, 
                        logger_file = LOGGER_SAVE_NAME)
    if mode == 'train':
        trainer.train(continue_epoch=previous_stop)
    elif mode == 'test':
        return trainer.evaluate()
    else:
        raise NotImplementedError()