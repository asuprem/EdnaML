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
```      
    Kwargs (MODEL_KWARGS):
        last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
        attention (str, None): The attention module to use. Only supports ['cbam', None]
        input_attention (bool, false): Whether to include the IA module
        ia_attention (bool, false): Whether to include input IA module
        part_attention (bool, false): Whether to include Part-CBAM Mobule
        secondary_attention (List[int], None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic block number provided here.

    Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS; set in backbones.resnet):
        zero_init_residual (bool, false): Whether the final layer uses zero initialization
        top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
        num_classes (int, 1000): Number of features in final imagenet FC layer
        groups (int, 1): Used during resnet variants construction
        width_per_group (int, 64): Used during resnet variants construction
        replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
        norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D
```

# `MODEL.MODEL_NORMALIZATION`

The normalization parameter for the final feature layer(s) of the model. One of:

- `bn` - Batch Normalization
- `gn` - Group Normalization
- `ln` - Layer Normalization
- `in` - Instance Normalization
- `l2` - L2 Normalization

# `MODEL.EMBEDDING_DIMENSIONS`

Whether to downscale feature dimensions further before projecting to softmax/classification layer. If `None`, ignored. Else, if there is an **Integer** value, then therer will be an additional FC layer with `EMBEDDING_DIMENSIONS` number of parameters between the feature layer and the final classification FC layer.


