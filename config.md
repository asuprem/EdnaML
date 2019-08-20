# Configuration

Each configuration file is a YAML file. A YAML file is formatted as `key: value` pairs. A key may have sub keys as follows:

    # Example taken from https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html
    # An employee record
    martin:
        name: Martin D'vloper
        job: Developer
        skill: Elite

This readme describes all configuration options. Each section represents a top-level key. Each subsection describes a sub-key. A subsection covers value descriptors for the key. You can take a look at some of the configurations in `/config/`.

- DATASET
    - ROOT_DATA_FOLDER: `str`. The folder containing training, testing, and query images.
    - TRAIN_FOLDER: `str`. The folder within ROOT_DATA_FOLDER with the training images
    - TEST_FOLDER: `str`. The folder within ROOT_DATA_FOLDER with the testing/gallery images
    - QUERY_FOLDER: `str`. The folder within ROOT_DATA_FOLDER with the query images
    - SHAPE: `array-like of int with shape 1x2`. Images will be resized to this shape.

- TRANSFORMATION
    - NORMALIZATION_MEAN: `float` or `array-like of float with shape 1x3`. Normalization mean parameter for image transformation
    - NORMALIZATION_STD: `float` or `array-like of float with shape 1x3`. Normalization standard deviation parameter for image transformation
    - NORMALIZATION_SCALE: `float`. Image pixel values will be scaled by 1/NORMALIZATION_SCALE. 
    - H_FLIP: `float between 0 and 1`. Probability of horizontal flip during training.
    - T_CROP: `bool`. Whether training augmentation should include random cropping.
    - RANDOM_ERASE: `bool`. Whether to include random erase augmentation, from [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)
    - RANDOM_ERASE_VALUE: `float`. Value to replace randomly erased region.
    - CHANNELS: `int`. Number of channels in image. Should be 3. Will be removed in future versions.
    - BATCH_SIZE: `int`. Number of images per batch.
    - INSTANCES: `int`. Number of images per ID in a batch. BATCH_SIZE should be divisible by INSTANCES.
    - WORKERS: `int`. Number of CPU threads to spawn for data loading. If you get pickling errors, reduce this to 1.

- MODEL
    - MODEL_ARCH: `str`. Model architecture to use. See section on Architecture for supported architectures.
    - MODEL_BASE: `str`. Model base for the given architecture. See section on Architecture for supported bases for each architecture
    - EMB_DIM: `int`. Dimensionality of feature embedding. Should be less than or equal to architecture output before fully connected layers, e.g. Resnet-50 outputs 1x2048 features.
    - MODEL_NORMALIZATION: `str`. Normalization to use for features. One of:
        1. '' (empty string) - No normalization
        2. 'l2' - L2 normalization
        3. 'bn' - Batch normalization
    - MODEL_KWARGS: `dict as json str`. Dict of model keyword arguments. See section on Architectures for kwargs for each architecture.

- SAVE
    - SAVE_FREQUENCY: `int`. Epoch to wait between model, optimizer, and scheduler backup.
    - MODEL_VERSION: `int`. Experiment version. useful if running the same experiment again.
    - MODEL_CORE_NAME: `str`. Name of experiment (or model). Used to name folders during saving.
    - MODEL_BACKBONE: `str`. Nickname for model backbone. Useful to keep track of multiple models/experiments. 
    - MODEL_QUALIFIER: `str`. Additional qualifier string. Useful if running same model on different datasets, etc.
    - DRIVE_BACKUP: `bool`. Whether to backup saves to another directory. Used for Google Drive backup in Colab.

- EXECUTION
    - MODEL_SERVING: `str`. Not used at the moment.
    - EPOCHS: `int`. Number of epochs to train.
    - TEST_FREQUENCY: `int`. Epochs to wait between evaluating model.

- LOSS
    - LOSSES: `list of str`. Losses to use in experiment. See section on Losses for list of supported losses.
    - LOSS_KWARGS: `list of dict`. Loss parameters. See section on Losses for loss parameters.
    - LOSS_LAMBDAS: `list of float`. Weights for each loss.

- OPTIMIZER
    - OPTIMIZER_NAME: `str`. Name of optimizer. All pytorch optimizers should work, but tested only with Adam. 
    - OPTIMIZER_KWARGS: `dict as json str`. Optimizer parameters that should be passed during initialization.
    - BASE_LR: `float`. Starting learning rate.
    - LR_BIAS_FACTOR: `float`. Multiplicative factor for learning rate for bias parameters within model.
    - WEIGHT_DECAY: `float`. Weight decay parameter.
    - WEIGHT_BIAS_FACTOR: `float`. Multiplicative factor for learning rate weight decay for bias parameters within model.
    - FP16: `bool`. Whether to use mixed precision traning. You will need Nvidia Apex for this. NVIDIA Apex is not supported on Windows yet.

- SCHEDULER
    - LR_SCHEDULER: `str`. Name of LR Scheduler. All pytorch schedulers are supported. See section on Schedulers for additional schedulers.
    - LR_KWARGS: `dict as json str`. LR scheduler parameters that should be passed during initialization.

- LOGGING
    - STEP_VERBOSE: `int`. Number of steps in a batch before logging loss and accuracy.
