# MODELS

The model section determines the architecture of the trained EdnaML model.



The `MODEL.MODEL_BUILDER` paramter controls the overall model research architecture base. For each model-builder, there are specific values for most of remaining parameters. We group them under the model-builder section. Common parameters are described afterwards.

Values for `MODEL.MODEL_BUILDER` include:

- `colabel_model_builder`

## `MODEL.MODEL_BUILDER: colabel_model_builder`

There are several architectures in `MODEL.MODEL_ARCH` for `colabel_model_builder`. For each architecture, we have different bases.
### `MODEL.MODEL_ARCH: CoLabelResnet`

For `CoLabelResnet`, the following are available:

- `MODEL_BASE`: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `MODEL_KWARGS`: see [`CoLabelResnetAbstract`](/models/abstracts.py) class

    - `last_stride` (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2. No need to change.
    - `attention` (str, None): The attention module to use. Supports [`'cbam'`, `'dbam'`]. This is applied to all layers, or to specific ResNet blocks, depending on the value of `secondary_attention`
    - `input_attention` (bool, false): Whether to include the IA module at the first ResNet block. You can only have either `input_attention`, or `ia_attention`, but not both.
    - `ia_attention` (bool, false): Whether to include input IA module at the first Conv Layer. You can only have either `input_attention`, or `ia_attention`, but not both.
    - `part_attention` (bool, false): Whether to include part attention module at the first ResNet block for. Cannot have `part_attention` and `attention`, unless `secondary_attention`>1.
    - `secondary_attention` (`1`|`2`|`3`|`4`|`None`): Modifies `attention` to apply it to specific Resnet blocks, instead of everywhere. `None` means `attention` is applied to all layers. `{1,...,4}` applies `attention` to the ResNet block specified.

    The following are default arguments <span style="color:magenta;font-weight:bold">(DO NOT CHANGE OR ADD TO MODEL_KWARGS)</span>; These are set during model creation.

        - `zero_init_residual` (bool, false): Whether the final layer uses zero initialization
        - `top_only` (bool, false): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        - `num_classes` (int, 1000): Number of features in final imagenet FC layer
        - `groups` (int, 1): Used during resnet variants construction
        - `width_per_group` (int, 64): Used during resnet variants construction
        - `replace_stride_with_dilation` (bool, None): Well, replace stride with dilation...
        - `norm_layer` (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D


## `MODEL.MODEL_BUILDER: multiclassification_model_builder`

Multiclassification is when the model has multiple outputs, and each output is for a different task. However, they all share the convolutional parameters. Example is a model that on the same backbone, performs both color and type detection.

There are several architectures in `MODEL.MODEL_ARCH` for `multiclassification_model_builder`. For each architecture, we have different bases.

There are also additional parameters:

- `NUMBER_OUTPUTS`: This is the number of classification outputs for the model
- `OUTPUT_DIMENSIONS`: This is the number of classes in each dimension. This can be left blank if you want the builder to automatically infer the dimensions using the number of classes in the dataset
- `OUTPUT_CLASSNAMES`: This is the name of each of the outputs. The names of outputs should correspond to the label names in `DATASET_ARGS.classificationclass` in `EXECUTION.DATAREADER`, but they do have have to be in the same order.
- `LABELNAMES`: This is the name of each label provided in a batch. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span>
### `MODEL.MODEL_ARCH: MultiClassificationResnet`

For `MultiClassificationResnet`, the following are available:

- `MODEL_BASE`: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `MODEL_KWARGS`: see [`MultiClassificationResnetAbstract`](/models/abstracts.py) class

    - `last_stride` (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2. No need to change.
    - `attention` (str, None): The attention module to use. Supports [`'cbam'`, `'dbam'`]. This is applied to all layers, or to specific ResNet blocks, depending on the value of `secondary_attention`
    - `input_attention` (bool, false): Whether to include the IA module at the first ResNet block. You can only have either `input_attention`, or `ia_attention`, but not both.
    - `ia_attention` (bool, false): Whether to include input IA module at the first Conv Layer. You can only have either `input_attention`, or `ia_attention`, but not both.
    - `part_attention` (bool, false): Whether to include part attention module at the first ResNet block for. Cannot have `part_attention` and `attention`, unless `secondary_attention`>1.
    - `secondary_attention` (`1`|`2`|`3`|`4`|`None`): Modifies `attention` to apply it to specific Resnet blocks, instead of everywhere. `None` means `attention` is applied to all layers. `{1,...,4}` applies `attention` to the ResNet block specified.

    The following are default arguments <span style="color:magenta;font-weight:bold">(DO NOT CHANGE OR ADD TO MODEL_KWARGS)</span>; These are set during model creation.

        - `zero_init_residual` (bool, false): Whether the final layer uses zero initialization
        - `top_only` (bool, false): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        - `num_classes` (int, 1000): Number of features in final imagenet FC layer
        - `groups` (int, 1): Used during resnet variants construction
        - `width_per_group` (int, 64): Used during resnet variants construction
        - `replace_stride_with_dilation` (bool, None): Well, replace stride with dilation...
        - `norm_layer` (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D





# `MODEL.MODEL_NORMALIZATION`

The normalization parameter for the final feature layer(s) of the model. One of:

- `bn` - Batch Normalization
- `gn` - Group Normalization
- `ln` - Layer Normalization
- `in` - Instance Normalization
- `l2` - L2 Normalization

# `MODEL.EMBEDDING_DIMENSIONS`

Whether to downscale feature dimensions further before projecting to softmax/classification layer. If `None`, ignored. Else, if there is an **Integer** value, then therer will be an additional FC layer with `EMBEDDING_DIMENSIONS` number of parameters between the feature layer and the final classification FC layer.


