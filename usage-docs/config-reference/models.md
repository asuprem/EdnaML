# MODELS

The model section determines the architecture of the trained EdnaML model.



The `MODEL.MODEL_BUILDER` paramter controls the overall model research architecture base. For each model-builder, there are specific values for most of remaining parameters. We group them under the model-builder section. Common parameters are described afterwards.

Values for `MODEL.MODEL_BUILDER` include:

- `classification_model_builder`

## `MODEL.MODEL_BUILDER: classification_model_builder`

There are several architectures in `MODEL.MODEL_ARCH` for `classification_model_builder`. For each architecture, we have different bases.
### `MODEL.MODEL_ARCH: ClassificationResnet`

For `ClassificationResnet`, the following are available:

- `MODEL_BASE`: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `MODEL_KWARGS`: see [`ClassificationResnetAbstract`](/models/abstracts.py) class

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

There are also additional parameters for `MODEL_KWARGS` common to all MultiClassification models

- `number_outputs`: This is the number of classification outputs for the model
- `outputs`: A list, each element is the i-th output's metadata
    - `dimensions`: This is the number of classes for this output. This can be left blank if you want EdnaML to infer the size from the `LABEL` parameter below. 
    - `name`: This is the name of this output. This is used by model_builder to keep track of output names and labels. If left blank, EdnaML will automatically name all outputs. Outputs are named because multiple outputs can track the same class, sometimes, e.g. in a contrastive non-weight sharing setting
    - `label`: This is the name of the label this output is tracking. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span> labels. 
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


## `MODEL.MODEL_BUILDER: multibranch_model_builder`

Multibranch is when a model has multiple branch, each with its own set of outputs. Branches do not share conv layers. There are more exotic multibranch models that *do* share layers and parameters, but we deal with the simple case here.

There are several architectures in `MODEL.MODEL_ARCH` for `multibranch_model_builder`. For each architecture, we have different bases.

There are also additional parameters for `MODEL_KWARGS` common to all MultiClassification models. **This are stricter for multi-branch models due to the many moving parts.**

- `number_branches`: This is the number of branches for the model
- `branches`: A list, each element is the i-th branch's metadata
    - `name`: This is the name of this branch. This is used by model_builder to keep track of output names and labels. 
    - `number_outputs`: This is the number of classification outputs for this branch
    - `outputs`: A list, each element is the j-th output's metadata for this branch
        - `dimensions`: This is the number of classes for this output. This can be left blank if you want EdnaML to infer the size from the `LABEL` parameter below. 
        - `name`: This is the name of this output. 
        - `label`: This is the name of the label this output is tracking. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span> labels.
- `fuse`: **Bool**. Whether branch outputs are going to be fused
- `fuse_outputs`: List of output names (not branch names) that are fused
- `fuse_dimensions`: The dimensions of the fused output. If left blank, EdnaML will infer from `fuse_label`
- `fuse_label`: The label that tracks the fused output




    - `dimensions`: This is the number of classes for this output. This can be left blank if you want EdnaML to infer the size from the `LABEL` parameter below. 
    
    - `label`: This is the name of the label this output is tracking. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span> labels. 
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


