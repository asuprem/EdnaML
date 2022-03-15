# EdnaML Configuration

EdnaML takes in configuration files that outline each step in the ML pipeline. We describe the sections here. For each section, there may be additional documentatation inside their respective config directory.

The configuration file is in YAML format. Each subsequent section is a top-level YAML key. See sample config file [here](config.yml).

Near the end, there is a <span style="color:teal; font-weight:bold">Potential Pitfalls</span> section.

# EXECUTION

The `EXECUTION` section contains parameters for EdnaML pipeline execution. They are:

- `OPTIMIZER_BUILDER`: The optimizer to choose. See [`optimizers.md`](config-reference/optimizers.md)
- `MODEL_SERVING`: unused.
- `EPOCHS`: The number of epochs to train the model. **Integer**
- `SKIPEVAL`: Whether to skip the initial evaluation stage. This is useful during debugging if the initial evaluation stage takes too long and is not useful. **Boolean**
- `TEST_FREQUENCY`: The evaluation frequency in epochs, e.g. every how many epochs to evaluate the model being trained. **Integer**
- `DATAREADER`: The datareader key. 
    - `DATAREADER`: The datareader class to use. See [`datareaders.md`](config-reference/datareaders.md).
    - `CRAWLER_ARGS`: A key-value dictionary or keyword arguments to pass into the datareader's crawler. Documentation is provided in [`datareaders.md`](config-reference/datareaders.md).
    - `DATASET_ARGS`: A key-value dictionary or keyword arguments to pass into the datareader's TorchDataset. Documentation is provided in [`datareaders.md`](config-reference/datareaders.md).
    - `GENERATOR_ARGS`: A key-value dictionary or keyword arguments to pass into the datareader's Generator. Documentation is provided in [`datareaders.md`](config-reference/datareaders.md).
    - `DATALOADER_ARGS`: A key-value dictionary or keyword arguments to pass into the datareader's TorchDataloader. Documentation is provided in [`datareaders.md`](config-reference/datareaders.md).


## Sample

```
EXECUTION:
  OPTIMIZER_BUILDER: CoLabelOptimizer
  MODEL_SERVING: None
  EPOCHS: 10
  TEST_FREQUENCY: 5
  DATAREADER: VehicleColor
    CRAWLER_ARGS: 
        ROOT_FOLDER: "Data/VehicleColors"
        TRAIN_FOLDER: "train"
        TEST_FOLDER: "test"
        VAL_FOLDER: "val"
    DATASET_ARGS:
        ANNOTATION: 1
    DATALOADER_ARGS:
```

# SAVE

The `SAVE` section contains parameters for model and results saving and backup options

- `MODEL_VERSION`: The current experiment version. **Integer**
- `MODEL_CORE_NAME`: The core name of the experiment
- `MODEL_BACKBONE`: Annotation of which backbone is used in this experiment
- `MODEL_QUALIFIER`: Any additional annotation we might need for this experiment
- `DRIVE_BACKUP`: Whether to backup intermediate models to a netwrk drive. **Boolean**
- `SAVE_FREQUENCY`: How often to save model, epoch-wise. **Integer**
- `CHECKPOINT_DIRECTORY`: The root network directory for backups.

## Sample

```
SAVE:
  MODEL_VERSION: 1
  MODEL_CORE_NAME: "colabel_devel"
  MODEL_BACKBONE: "res18"
  MODEL_QUALIFIER: "all"
  DRIVE_BACKUP: False
  SAVE_FREQUENCY: 5 # Epoch
  CHECKPOINT_DIRECTORY: drive/MyDrive/Projects/CoLabeler/Models/"
```

# TRANSFORMATION

The `TRANSFORMATION` section contains parameters for data transformations and augmentations. A list of all `TRANSFORMATION` parameters is shown in [`transformations.md`](config-reference/transformations.md). The common ones are given below.

- `SHAPE`: The image shape to resize, if working with image or video data. **Array**. 
- `NORMALIZATION_MEAN`: The image normalization mean parameter to use. **Float**
- `NORMALIZATION_STD`: The image normalization standard deviation parameter to use. **Float** 
- `NORMALIZATION_SCALE`: The image normalization scale parameter to use. Unused in code. **Float**
- `CHANNELS`: Number of channels in image. **Integer**
- `BATCH_SIZE`: The size of batches during traning. **Integer**
- `WORKERS`: The number of parallel workers for data loading. **Integers**


# MODEL

The `MODEL` section contains parameters to build a model. Additional details are provided in [`models.md`](config-reference/models.md)


- `MODEL_ARCH`: The architecture class to use, from `EdnaML.models`. See [`models.md`](config-reference/models.md).
- `MODEL_BASE`: The base parameter for the provided `MODEL_ARCH`. See [`models.md`](config-reference/models.md).
- `EMB_DIM`: The number of embedding dimensions. Potentially deprecated. 
- `MODEL_NORMALIZATION`: The feature normalization parameters for output features. See [`models.md`](config-reference/models.md).
- `MODEL_KWARGS`: A key-value dictionary of any additional `MODEL_ARCH` specific parameters. These may differ between convnets, transformers, rnns, etc. See [`models.md`](config-reference/models.md).
- `MODEL_TRAINING_OUTPUTS`: Integer corresponding to number of model outputs during training. Useful for multi-output models. This should also correspond to the tuples yielded by the crawler. Used for loss function, potentially. 
- `MODEL_TRAINING_OUTPUT_NAMES`: Array of model training output names. Size of array must be equal to `MODEL_TRAINING_OUTPUTS`

# LOSS

The `LOSS` section contains parameters to set up a model's loss function. Additional details are provided in ['loss.md'](config-reference/loss.md).

It is an array of the different losses used. Each entry in the array corresponds to the loss for one output. The array entries are as follows:

- `LOSSES`: An array of loss function classes to use. See ['loss.md'](config-reference/loss.md)
- `LOSS_KWARGS`: An array of arguments for each loss used. See ['loss.md'](config-reference/loss.md)
- `LOSS_LAMBDAS`: The weight for each loss function. The weights are normalized from here.



# OPTIMIZER

The `OPTIMIZER` section contains parameters for the training optimizer. These are, inside ednaml, fed into the `EXECUTION.OPTIMIZER_BUILDER`. Additional details foud in [`optimizers.md`](config-reference/optimizers.md)

  `OPTIMIZER_NAME`: The name for the optimizer to use in the builder. Usually "Adam". See [`optimizers.md`](config-reference/optimizers.md)
  `OPTIMIZER_KWARGS`: Any additional arguments for the optimizer function.
  `BASE_LR`: The starting learning rate. 
  `LR_BIAS_FACTOR`: The learning rate bias factor. Usually `1.0`
  `WEIGHT_DECAY`: Weight decay parameters. Usually `0.0005`
  `WEIGHT_BIAS_FACTOR`: Weight bias factor. Usually `0.0005`
  `FP16`: Whether to use mixed precision. Unused. **Boolean**

## Sample

Here is a sample for losses for 2 outputs:

```
LOSS:
  - LOSSES: ['SoftmaxLogitsLoss']
    KWARGS: [{}]
    LAMBDAS: [1.0]
    NAME: out1
  - LOSSES: ['SoftmaxLogitsLoss', 'CenterLoss']
    KWARGS: [{}, {"center_loss_args":"center_loss_values"}]
    LAMBDAS: [0.5, 0.5]
```

# SCHEDULER

The `SCHEDULER` section contains parameter for the learning rate scheduler in any learnable parameter. For now, it is a single scheduler across the entire training scope.


- `LR_SCHEDULER`: The name of the scheduler. See ['schedulers.md'](config-reference/schedulers.md)
- `LR_KWARGS`: The parameters for this scheduler. See ['schedulers.md'](config-reference/schedulers.md)


# LOGGING

The `LOGGING` section contains parameters for any logging during experiment runs.

- `STEP_VERBOSE`: Per how many training steps to output any loss or other recording parameters.