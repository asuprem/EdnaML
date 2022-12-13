# Contents

- [Contents](#contents)
- [EdnaML Design](#ednaml-design)
  - [Core Abstractions](#core-abstractions)
- [EdnaML Configuration](#ednaml-configuration)
  - [`EXECUTION`](#execution)
    - [Example:](#example)
  - [`DEPLOYMENT`](#deployment)
  - [`DATAREADER`](#datareader)
    - [Example](#example-1)
  - [`STORAGE`](#storage)
    - [Example](#example-2)
  - [`SAVE`](#save)
    - [Example](#example-3)
  - [`TRANSFORMATION`](#transformation)
    - [Example](#example-4)
  - [`TRAIN_TRANSFORMATION`](#train_transformation)
    - [Example](#example-5)
  - [`TEST_TRANSFORMATION`](#test_transformation)
    - [Example](#example-6)
  - [`MODEL`](#model)
    - [Example](#example-7)
  - [`LOSS`](#loss)
    - [Example](#example-8)
  - [`OPTIMIZER`](#optimizer)
    - [Example](#example-9)
  - [`SCHEDULER`](#scheduler)
  - [`MODEL_PLUGIN`](#model_plugin)
    - [Example](#example-10)
  - [`LOGGING`](#logging)

# EdnaML Design

EdnaML formalizes ML pipelines for reproducibility by providing users a high-level declarative API and structured pipeline design and deployment patterns.

A basic EdnaML pipeline can be executed with only 4 lines:

```
from ednaml.core import EdnaML
eml = EdnaML(config = [config1, config2, config3...])
eml.apply()
eml.train()
```

EdnaML abstracts much of the boilerplate code in pipeline design, model building, loss building, optimizer setup, logging, saving models and artifacts, provenance, and reproducibility, leaving you to do actual experiments! That said, EdnaML also exposes the entire stack for bespoke pipelines with intelligent callbacks, decorators, and configuration options. Further, these customizations are also tracked with EdnaML's provenance management, allowing for reproducibility even when using third party tools and code.

## Core Abstractions

EdnaML's core abstraction of a pipeline is decomposed into the basic stages/components of any ML or data analytics pipeline:
    - Data ingest (ednaml.crawlers)
    - Cleaning and processing (ednaml.generators)
    - Model design (ednaml.models)
    - Machine Learning (ednaml.optimizer, ednaml.scheduler, ednaml.loss)
    - Training and Evaluation (ednaml.trainers)
    - Monitoring callbacks and KPIs (ednaml.plugins)
    - Deployment (ednaml.deploy)
    - Storage of models, metrics, configurations, logs (ednaml.storage)

Each of these stages/components is specified for an experiment with configuration files, along with sensible default options. A full configuration is shown in [config-full.yml](config-full.yml). At this point, we would recommend following some of the examples and samples first to get an idea of how EdnaML works, before jumping into the configuration details.


# EdnaML Configuration

## `EXECUTION`

The `EXECUTION` section contains configuration options for an EdnaML pipeline. EdnaML pipelines are used to train and evaluate learnable models. It contains the following options:

| Key               | Type                                | Default Value             | Notes                                                                                                        |
| ----------------- | ----------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| TRAINER           | string                              | `BaseTrainer`             | Any trainer that inherits from `ednaml.trainers.BaseTrainer`. Usually, `BaseTrainer` is sufficient.                                                 |
| TRAINER_ARGS      | dict[str, any]                      | `{accumulation_steps: 1}` | Any arguments that need to be passed to the `TRAINER`                                                        |
| OPTIMIZER_BUILDER | string                              | `ClassificationOptimizer` | Any optimizer builder that inherits from `ClassificationOptimizer`. <br> Likely does not need to be changed. |
| MODEL_SERVING     | string                              | "Unused"                  | Unused for now                                                                                               |
| EPOCHS            | int                                 | 10                        | The number of epochs to train for. This value is provided to the `TRAINER`                                   |
| SKIPEVAL          | bool                                | `False`                   | Whether to skip the initial evaluation. This value is provided to the `TRAINER`                              |
| TEST_FREQUENCY    | int                                 | 1                         | How often, in epochs, to evaluate the model using the `TRAINER`'s `evaluate()` method                        |
| FP16              | bool                                | `False`                   | Whether to use FP16 mode (half-precision) or full pecision. Currently unsupported.                           |
| PLUGIN            | --                                  | --                        | This is unneeded for EXECUTION, but remains to support some legacy code.                                     |
| PLUGIN.HOOKS      | "always"<br>"warmup"<br>"activated" | "always"                  | Unneeded                                                                                                     |
| PLUGIN.RESET      | bool                                | `False`                   | Unneeded                                                                                                     |
|                   |                                     |                           |                                                                                                              |


### Example:

```
EXECUTION:
  EPOCHS: 5                         # Execute for 5 epochs
  TEST_FREQUENCY: 1                 # Test the model every 1 epoch
  TRAINER: ClassificationTrainer    # Use built-in ClassificationTrainer
  TRAINER_ARGS: 
    accumulation_steps: 2           # Pass `accumulation_steps=2` into ClassificationTrainer
```

`OPTIMIZER_BUILDER` is defaulted to `ClassificationOptimizer`, since it is not provided here. 



## `DEPLOYMENT`

The `DEPLOYMENT` section contains configuration options for an EdnaDeploy pipeline. EdnaDeploy pipeline are used to deploy trained ML models, or statistical models, or any data processing step that does not need learning. It is similar to `EXECUTION`, with a few key differences:

1. `TRAINER` -> `DEPLOY` ; `TRAINER_ARGS` -> `DEPLOYMENT_ARGS`
2. New key `OUTPUT_ARGS`
3. No `SKIPEVAL`, `TEST_FREQUENCY`, `FP16`, `OPTIMIZER_BUILDER`


| Key             | Type                                | Default Value | Notes                                                                                                                                                                                                                                              |
| --------------- | ----------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DEPLOY          | string                              | `BaseDeploy`  | Any deploy that inherits from `ednaml.deploy.BaseDeploy`.                                                                                                                                                                                          |
| DEPLOYMENT_ARGS | dict[str, any]                      | `{}`          | Any arguments that need to be passed to the `DEPLOY`, specifically for constructing the object.                                                                                                                                                    |
| EPOCHS          | int                                 | 1             | The number of epochs to run the deployment for. This value is provided to the `DEPLOY`                                                                                                                                                             |
| OUTPUT_ARGS     | dict[str, any]                      | `{}`          | Any arguments that need to be passed to the `DEPLOY`, specifically for outputting results                                                                                                                                                          |
| PLUGIN          | --                                  | --            | Suboptions control how plugins, which are bespoke monitoring or <br> preprocessing functionality added to models in an ad-hoc fashion, are used.                                                                                                   |
| PLUGIN.HOOKS    | "always"<br>"warmup"<br>"activated" | "always"      | How often to fire plugins. `always` means plugins for for every batch. <br> `warmup` means plugins fire only during `warmup`. <br> `activated` means plugins fire only after they have warmed up. <br> See additional details in `PLUGIN` section. |
| PLUGIN.RESET    | bool                                | `False`       | Whether to reset already generated and saved plugins, potentially replacing them. Unused.                                                                                                                                                          |
|                 |                                     |               |                                                                                                                                                                                                                                                    |


## `DATAREADER`

A `DATAREADER` is a high-level abstraction for data processing, implemented in `ednaml.datareaders`. DataReaders
contain 3 subcomponents: (i) a Crawler (implemented in `ednaml.crawlers`), (ii) a Generator inheriting from torch's data loaders 
(EdnaML contains sensible base classes for TextGenerator and ImageGenerator in `ednaml.generators`), and (iii) a Dataset class
(also in `ednaml.generators) that inherits from Torch's Dataset class.

The DataReader classes contain default Crawler, Generator, and Dataset for common tasks such as image and text classification. However, in most cases,
you will need to overwrite the classes to fit your use case. Often, you only need to overwrite the default `Crawler`. You can pass in a custom `Crawler` class
with `ednaml`'s decorators; we show several examples in the `sample-configs` directory.



| Key            | Type                                | Default Value | Notes                                                                                                                                                                                                                                              |
| -------------- | ----------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DATAREADER     | string                              | `DataReader`  | The base DataReader class uses only abstract classes. You should replace this <br> with sensible options, such as `TorchVisionDatareader` to use Torchvision datasets. <br> See samples for more examples.                                         |
| CRAWLER_ARGS   | dict[str, any]                      | `{}`          | Any arguments that need to be passed to the `CRAWLER`'s `__init__()`. <br> This applies to overridden `CRAWLER`s as well.                                                                                                                          |
| DATASET_ARGS   | dict[str, any]                      | `{}`          | Any arguments that need to be passed to the `DATASET`'s `__init__()`. <br> This applies to overridden `DATASET`s as well. `DATASET` are usually instantiated inside a `GENERATOR`.                                                                 |
| GENERATOR_ARGS | dict[str, any]                      | `{}`          | Any arguments that need to be passed to the `GENERATOR`. <br> This applies to overridden `GENERATOR`s as well.                                                                                                                                     |
| GENERATOR      | string                              | `null`        | If overriding the default `GENERATOR` from the provided `DataReader` class.                                                                                                   |
|                |                                     |               |                                                                                                                                                                                                                                                    |

### Example

The following example sets up a pipeline's DataReader to load MNIST. 

```
DATAREADER: 
  DATAREADER: TorchvisionDatareader   # We will use the built-in Torchvision Datareader to download and load MNIST
  GENERATOR_ARGS:
    tv_dataset: MNIST                 # This tells our datareader that we want MNIST. Argument details are usually found within DataReader's documentation.
    tv_args: 
      root: "Data/"                     # This tells where to download the dataset
      args:
        download: true
  DATASET_ARGS:
    label_name: mnist_digits
```

## `STORAGE`

This section sets up storage backends for recording training and deployment artifacts. To actually upload artifacts, we still need to enable backup in the `SAVE` section.

The `STORAGE` top level key contains a list of storages (see example below). Each entry has the following options. 

**NOTE: There are NO default options. The default is an empty Storage list, meaning nothing will be backed up. This is useful for ephemeral experiments where a backup is not desired.**

| Key           | Type           | Default Value | Notes                                                                                                                           |
| ------------- | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| STORAGE_NAME  | string         | No Default    | A name for this storage to keep track of it. <br> It can be any alphanumeric sequence, for identification purposes only.        |
| STORAGE_CLASS | string         | No Default    | The type of Storage, from `ednaml.storage`, <br> e.g. `LocalStorage`, `AzureStorage`, `MlFlowStorage`, etc, or a custom Storage |
| STORAGE_URL   | string         | No Default    | An argument passed into the Storage instantiation <br> that is common to all Storages. Usually the location of the Storage.     |
| STORAGE_ARGS  | dict[str, any] | No Default    | Any other arguments that are necessary for the this specific Storage class.                                                     |
|               |                |               |                                                                                                                                 |

### Example

The following sets up a single storage, using `LocalStorage`, where supported and enabled artifacts will be backed up to the directory called `./backup` .

```
STORAGE:
  - STORAGE_NAME: storage-1 
    STORAGE_CLASS: LocalStorage
    STORAGE_URL: ./backup
    STORAGE_ARGS: {}
```

## `SAVE`

This manages the identifier for the experiment, as well as backup options. 

| Key                 | Type   | Default Value            | Notes                                                                                                                           |
| ------------------- | ------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| MODEL_VERSION       | int    | 1                        | The version for this experiment. Increment for related experiments using different architectures, for example.                  |
| MODEL_CORE_NAME     | string | "model"                  | The name for this experiment. Should be something relatively unique.                                                            |
| MODEL_BACKBONE      | string | "backbone"               | The backbone for this experiment. Can also be the tool or approach. Essentially an identifier.                                  |
| MOEDL_QUALIFIER     | string | "all"                    | Yet another qualifier. Usually refers to dataset or some narrowing of scope for this experiment.                                |
| BACKUP              | --     | --                       | This is a top level key of generic backup options that apply to all generated artifacts                                         |
| BACKUP.BACKUP       | bool   | `False`                  | Whether to back up.                                                                                                             |
| BACKUP.STORAGE_NAME | string | "reserved-empty-storage" | A Storage defined in `STORAGE` to use for backups. <br>`reserved-empty-storage` is a built-in Storage that doesn't do anything. |
| FREQUENCY           | int    | 0                        | How often, in epochs, to back up                                                                                                |
| FREQUENCY_STEP      | int    | 0                        | How often, in number of steps inside epoch, to back up.                                                                         |
| LOB_BACKUP          | --     | --                       | A top level key controlling log backups. Same options as `BACKUP`                                                               |
| CONFIG_BACKUP       | --     | --                       | A top level key controlling configuration backups. Same options as `BACKUP`                                                     |
| MODEL_BACKUP        | --     | --                       | A top level key controlling model backups. Same options as `BACKUP`                                                             |
| ARTIFACTS_BACKUP    | --     | --                       | A top level key controlling training artifact backups. If not provided, fall back on `MODEL_BACKUP`. Same options as `BACKUP`   |
| PLUGIN_BACKUP       | --     | --                       | A top level key controlling plugin backups. If not provided, fall back on `MODEL_BACKUP`. Same options as `BACKUP`              |
| METRICS_BACKUP      | --     | --                       | A top level key controlling metrics backups. Same options as `BACKUP`                                                           |
|                     |        |                          |                                                                                                                                 |


### Example

The following example creates an experiment with the name `<mnist_resnet, res18, mnist, v1>`.

The following artifacts are backed up to the storage with `STORAGE_NAME = local` every 200 steps within an epoch, as well as every epoch: `[model, model_artifacts, metrics, plugins, code]`

The following artifacts are NOT backed up at all: `[config]`.

The following artifacts are backed up every 5 epochs to a storage with `STORAGE_NAME = logstorage`: `[log]`.

```
SAVE:
  MODEL_VERSION: 1              # We are only running v1 of this experiment
  MODEL_CORE_NAME: mnist_resnet # The core name for this model
  MODEL_BACKBONE: res18         # What backbone we are using here
  MODEL_QUALIFIER: mnist        # Any other details we want to capture in the name
  BACKUP: 
    BACKUP: True
    STORAGE_NAME: local
    FREQUENCY: 1
    FREQUENCY_STEP: 200
  CONFIG_BACKUP:
    BACKUP: False
  LOG_BACKUP:
    STORAGE_NAME: logstorage
    FREQUENCY: 5
    FREQUENCY_STEP: 0
```


## `TRANSFORMATION`

This manages the transformations applied to data generation process during training and testing/deployment. May be unused in some generators.

| Key        | Type           | Default Value | Notes                                                                              |
| ---------- | -------------- | ------------- | ---------------------------------------------------------------------------------- |
| BATCH_SIZE | int            | 32            | The number of samples in each batch                                                |
| WORKERS    | int            | 2             | The number of parallel workers to use during data loading                          |
| ARGS       | dict[str, any] | `{}`          | Any arguments for the `GENERATOR`'s `build_transformation()` method. See examples. |
|            |                |               |                                                                                    |

### Example

```
TRANSFORMATION:
  BATCH_SIZE: 32                # The size of batches to provide to the model during training
  WORKERS: 2                    # The number of workers to use during training
  ARGS:                         # Additional args for this Generator (e.g. ednaml.generators.TorchvisionGeneratorWrapper)
    i_shape: [28,28]
    normalization_mean: 0.1307
    normalization_std: 0.3081
    normalization_scale: 0.5
    random_erase: False
    random_erase_value: 0.5
    channels: 1
```

## `TRAIN_TRANSFORMATION`

This overrides `TRANSFORMATION` for training data in EdnaML. Not all keys need to be provided. Only provided keys will override. Exception are `ARGS`; if `ARGS` is provided, it will completely overwrite any `TRANSFORMATION.ARGS`

| Key        | Type           | Default Value | Notes                                                                              |
| ---------- | -------------- | ------------- | ---------------------------------------------------------------------------------- |
| BATCH_SIZE | int            | 32            | The number of samples in each batch                                                |
| WORKERS    | int            | 2             | The number of parallel workers to use during data loading                          |
| ARGS       | dict[str, any] | `{}`          | Any arguments for the `GENERATOR`'s `build_transformation()` method. See examples. |
|            |                |               |                                                                                    |


### Example

If the following snippet is used in conjunction with the snippet in `TRANSFORMATION` above, then batch size and workers remain at 32 and 2, respectively. ARGS are completely replaced.

```
TRAIN_TRANSFORMATION:           # Any replacements for the generator during training
  ARGS:
    h_flip: 0.5
    t_crop: True
```

## `TEST_TRANSFORMATION`

This overrides `TRANSFORMATION` for testing data or for deployments in EdnaML. Not all keys need to be provided. Only provided keys will override. Exception are `ARGS`; if `ARGS` is provided, it will completely overwrite any `TRANSFORMATION.ARGS`

| Key        | Type           | Default Value | Notes                                                                              |
| ---------- | -------------- | ------------- | ---------------------------------------------------------------------------------- |
| BATCH_SIZE | int            | 32            | The number of samples in each batch                                                |
| WORKERS    | int            | 2             | The number of parallel workers to use during data loading                          |
| ARGS       | dict[str, any] | `{}`          | Any arguments for the `GENERATOR`'s `build_transformation()` method. See examples. |
|            |                |               |                                                                                    |

### Example

```
TEST_TRANSFORMATION:            # Any replacements for the generator during testing
  ARGS:
    h_flip: 0
```

## `MODEL`

This sets up the model architecture to be used. If a custom model is passed, `BUILDER` and `MODEL_ARCH` are not necessary.


| Key                 | Type           | Default Value          | Notes                                                                                                                                                                                                                                      |
| ------------------- | -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BUILDER             | string         | `ednaml_model_builder` | A model builder to use. Usually `ednaml_model_builder` suffices, <br> though there are specialized builders for GANs and other architectures.                                                                                              |
| MODEL_ARCH          | string         | ModelAbstract          | The architecture to use for the model, e.g. `ClassificationResnet`, or `HFAutoModel` (HuggingFace). <br> ModelAbstract is an abstract class and will throw errors.                                                                         |
| MODEL_BASE          | string         | `base`                 | The architecture base, for generic models. Used where supported. <br> For example `ClassificationResNet` supports `resnet18`, `resnet34`, etc. <br> `HFAutoModel` supports `albert-base-v2`  and other transformer bases from HuggingFace. |
| MODEL_NORMALIZATION | string         | `bn`                   | Normalization for end-to-end models, where supported. Most models do not need this.                                                                                                                                                        |
| MODEL_KWARGS        | dict[str, any] | `{}`                   | Any arguments for the model's construction.                                                                                                                                                                                                |
| PARAMETER_GROUPS    | list[str]      | `[opt-1]`              | Parameter groups inside the model, for simultaneous learning objectings. <br> The default suffices for most models. GANs may use multiple parameter groups.                                                                                |
|                     |                |                        |                                                                                                                                                                                                                                            |


### Example

```
MODEL:                              # These are model-specific details
  BUILDER: ednaml_model_builder     # ednaml_model_builder is a basic model_builder that verifies model arguments make sense
  MODEL_ARCH: ClassificationResnet  # The built-in architecture we are using
  MODEL_BASE: resnet18              # The base we are using for the architecture. For example, ClassificationResnet accepts resnet18, resnet34, resnet50, etc
  MODEL_KWARGS:
    initial_channels: 1             # Since MNIST is black and white with 1 channel, we use the `initial_channels` parameter of ClassificationResnet to set this
```

## `LOSS`

This manages the losses used in model training. The configuratione expects a list of Losses. The default is an EMPTY LIST. Each entry has the following keys.


| Key              | Type                | Default Value | Notes                                                                                                                                                                             |
| ---------------- | ------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LOSSES           | list[string]        | []            | A list of loss names to use, from built-in or custom losses provided through decorators.                                                                                          |
| KWARGS           | list[dict[str,any]] | []            | The arguments for instanting each loss in `LOSSES`.                                                                                                                               |
| LAMBDAS          | list[int]           | []            | The weights for each loss term in this loss list.                                                                                                                                 |
| LABEL            | string              | ""            | A label from the `CRAWLER` that this loss is targeting. Useful if you want EdnaML to automatically train. See examples for more information.                                      |
|                  |                     |               |                                                                                                                                                                                   |

### Example
The following sets up 2 losses. The first loss targets the `color` label of the data, and the second targets the `type` label.

The `colorloss` Loss, targeting the `color` label of the data, consists of 2 losses summed together, with weights 1 and 0.5.

The `somename` Loss, targeting the `vtype` label of the data, consists of a single loss fuction.

```
LOSS:
  - LOSSES: ['SoftmaxLogitsLoss', 'SoftmaxLabelSmooth']
    KWARGS: [{}, {"eps": 0.2}]
    LAMBDAS: [1.0, 0.5]
    LABEL: color
    NAME: colorloss
  - LOSSES: ['SoftmaxLogitsLoss']
    KWARGS: [{}]
    LAMBDAS: [1.0]
    LABEL: vtype
    NAME: somename
```

## `OPTIMIZER`

This section sets up the learning optimizer. It also expects a list of optimizers. Usually one suffices for most training pipelines. The default is a ClassificationOptimizer instance, with the following values.



| Key                | Type          | Default Value | Notes                                                                                                      |
| ------------------ | ------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| OPTIMIZER_NAME     | string        | opt-1         | The name for this optimizer. This MUST  correspond with an entry in the list from `MODEL.PARAMETER_GROUPS` |
| OPTIMIZER          | string        | Adam          | An optimizer, from `torch.optim`                                                                           |
| OPTIMIZER_KWARGS   | dict[str,any] | `{}`          | The arguments for this optimizer.                                                                          |
| BASE_LR            | float         | 0.00001       | The base learning rate for this optimizer.                                                                 |
| LR_BIAS_FACTOR     | float         | 1.0           | The bias factor for this optimizer's learning rate.                                                        |
| WEIGHT_DECAY       | float         | 5e-4          | The weight decay for this optimizer.                                                                       |
| WEIGHT_BIAS_FACTOR | float         | 5e-4          | The bias factor for this optimizer's weight decay.                                                         |
|                    |               |               |                                                                                                            |



### Example

The following sets up an `AdamW` optimizer using the default `opt-1` name to track the default `opt-1` parameter group of the corresponding model.

```
OPTIMIZER:
  - OPTIMIZER: AdamW
    OPTIMIZER_KWARGS: {}
    BASE_LR: 1.0e-3
    LR_BIAS_FACTOR: 1.0
    WEIGHT_DECAY: 0.0005
    WEIGHT_BIAS_FACTOR: 0.0005
```

## `SCHEDULER`

The Scheduler manages learning rate scheduling. It also expects a list. The default is `torch.schedulers.StepLR`, with decay every 20 epochs.

| Key                | Type          | Default Value      | Notes                                                                                                      |
| ------------------ | ------------- | ------------------ | ---------------------------------------------------------------------------------------------------------- |
| SCHEDULER_NAME     | string        | opt-1              | The name for this scheduler. This MUST  correspond with an entry in the list from `MODEL.PARAMETER_GROUPS` |
| LR_SCHEDULER       | string        | StepLR             | A scheduler, from `torch.schedulers`                                                                       |
| LR_KWARGS          | dict[str,any] | `{"step_size":20}` | The arguments for this scheduler.                                                                          |
|                    |               |                    |                                                                                                            |



## `MODEL_PLUGIN`

A plugin is any additional functionality attached to a trained classifier or deployed function. Examples include online cluster or drift / change detectors that monitor incoming samples and compare them to an in-memory history. 

This are used only in EdnaDeploy, not in EdnaML. The default is an empty list. The folliwing are default values for list entries, if not provided. This should be avoided, as the default values can cause instability.


| Key           | Type          | Default Value | Notes                                                                                                         |
| ------------- | ------------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| PLUGIN_NAME   | string        | mp-1          | The name for this plugin. Used in saving the plugin and restoring values from prior saves.                    |
| PLUGIN        | string        | ModelPlugin   | The class for this plugin. Use either a built-in class or pass a custom class with decorators (see examples). |
| PLUGIN_KWARGS | dict[str,any] | `{}`          | The arguments for this plugin.                                                                                |
|               |               |               |                                                                                                               |

### Example

The following example sets up a KMeansProxy [1] and a LogitConfidence [2] plugin. KMP clusters training data into a KDTree, and LogitConfidence is a reject option that compares logits-based confidence of predictions to the average logit-confidence of training data.

```
  - PLUGIN: FastKMeansProxy
    PLUGIN_NAME: FastKMP-cos
    PLUGIN_KWARGS:
      proxies: 20
      dimensions: 768
      dist: cosine
      iterations: 30
      batch_size: 256
      alpha: 0.6
      classifier_access: encoder.classifier
  - PLUGIN: LogitConfidence
    PLUGIN_NAME: logit-confidence
    PLUGIN_KWARGS:
      num_classes: 2
```

[1] Abhijit Suprem and Calton Pu (2022). Evaluating Generalizability of Fine-Tuned Models for Fake News Detection. IEEE CIC 2022.
[2] Abhijit Suprem and Calton Pu (2022). MiDAS: Multi-integrated Domain Adaptive Supervision for Fake News Detection. CoRR.


## `LOGGING`

The Logging section provides configuration option for a variety of miscellaneous options not yet categorized, including logging options.

| Key           | Type          | Default Value | Notes                                                                                                     |
| ------------- | ------------- | ------------- | --------------------------------------------------------------------------------------------------------- |
| STEP_VERBOSE  | int           | 100           | How often to print classification loss to logs, and how often to check for backup triggers. See examples. |
| INPUT_SIZE    | array         | Null          | The Input Size to use when estimating model memory footprint.                                             |
|               |               |               |                                                                                                           |





