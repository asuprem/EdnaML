from ednaml.utils import locate_class
from ednaml.utils.LabelMetadata import LabelMetadata


def classification_model_builder(
    arch,
    base,
    weights=None,
    normalization=None,
    metadata: LabelMetadata = None,
    parameter_groups=None,
    **kwargs
):
    """Corroborative/Colaborative/Complementary Labeler Model Builder

    This builds a model for colabeler. Refer to paper [] for general construction. The model contains:
        * Architecture core
        * Convolutional attention
        * Spatial average pooling
        * FC-Layer for embedding dimensionality change
        * Normalization layer
        * Softmax FC Layers

    Args:
        arch (str): Ther architecture to use. The string "Base" is added after architecture (e.g. "Resnet", "Inception"). Only "Resnet" is currently supported.
        base (str): The architecture subtype, e.g. "resnet50", "resnet18"
        weights (str): Local path to weights file for the architecture core, e.g. pretrained resnet50 weights path.
        normalization (str): Normalization layer for reid-model. Can be None. Supported normalizations: ["bn", "l2", "in", "gn", "ln"]
    
    KWARGS (dict):
        embedding_dimension (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions
        metadata: dict of class-name: num-classes to infer size of softmax dimensions.


    Returns:
        Torch Model: A Torch classification model

    """
    if arch != "ClassificationResnet":
        raise NotImplementedError()
    archbase = locate_class(subpackage="models", classpackage=arch, classfile=arch)
    # Extract the dimensions from the classes metadata provided by EdnaML
    kwargs["softmax_dimensions"] = metadata.getLabelDimensions()

    model = archbase(
        base=base,
        weights=weights,
        normalization=normalization,
        metadata=metadata,
        parameter_groups=parameter_groups,
        **kwargs
    )
    return model


def multiclassification_model_builder(
    arch, base, weights=None, normalization=None, metadata=None, parameter_groups=None, **kwargs
):
    """Multiclassification model builder. This builds a model with a single backbone, and multiple classification FC layers.

    The model contains:
        * Architecture core
        * Convolutional attention
        * Spatial average pooling
        * Normalization layer
        * Softmax FC Layers

    Args:
        arch (str): Ther architecture to use. The string "Base" is added after architecture (e.g. "MultiClassificationResnet", "MultiClassificationInception"). Only "MultiClassificationResnet" is currently supported.
        base (str): The architecture subtype, e.g. "resnet50", "resnet18"
        weights (str): Local path to weights file for the architecture core, e.g. pretrained resnet50 weights path.
        normalization (str): Normalization layer for reid-model. Can be None. Supported normalizations: ["bn", "l2", "in", "gn", "ln"]
        metadata (Dict[str, int]): Label names to num-classes dictionary

    Kwargs:
        - `number_outputs`: This is the number of classification outputs for the model
        - `outputs`: A list, each element is the i-th output's metadata
            - `dimensions`: This is the number of classes for this output. This can be left blank if you want EdnaML to infer the size from the `LABEL` parameter below. 
            - `name`: This is the name of this output. This is used by model_builder to keep track of output names and labels. If left blank, EdnaML will automatically name all outputs. Outputs are named because multiple outputs can track the same class, sometimes, e.g. in a contrastive non-weight sharing setting
            - `label`: This is the name of the label this output is tracking. <span style="color:magenta; font-weight:bold">THIS SHOULD CORRESPOND EXACTLY WITH `DATASET_ARGS.classificationclass`</span> labels. 
    Returns:
        Torch Model: A Torch multiclassification model

    """

    # make the MultiClassificationResnet, with resnet base, with multiple output fc layers
    if arch != "MultiClassificationResnet":
        raise NotImplementedError()
    archbase = locate_class(subpackage="models", classpackage=arch, classfile=arch)

    model = archbase(
        base=base,
        weights=weights,
        normalization=normalization,
        metadata=metadata,
        parameter_groups=parameter_groups,
        **kwargs
    )
    return model


def multibranch_model_builder(
    arch, base, weights=None, normalization=None, metadata=None, parameter_groups=None,**kwargs
):
    """Multibranch model builder builds a model with multiple branches, each with their set of outputs.

    The model contains:
        * Architecture core
        * Convolutional attention
        * Spatial average pooling
        * Normalization layer
        * Softmax FC Layers

    Args:
        arch (str): Ther architecture to use. The string "Base" is added after architecture (e.g. "MultiClassificationResnet", "MultiClassificationInception"). Only "MultiClassificationResnet" is currently supported.
        base (str): The architecture subtype, e.g. "resnet50", "resnet18"
        weights (str): Local path to weights file for the architecture core, e.g. pretrained resnet50 weights path.
        normalization (str): Normalization layer for reid-model. Can be None. Supported normalizations: ["bn", "l2", "in", "gn", "ln"]
        metadata (Dict[str, int]): Label names to num-classes dictionary

    Kwargs:
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

    Returns:
        Torch Model: A Torch Re-ID model

    """

    # make the MultiClassificationResnet, with resnet base, with multiple output fc layers
    if arch != "MultiBranchResnet":
        raise NotImplementedError()
    archbase = locate_class(subpackage="models", classpackage=arch, classfile=arch)


    model = archbase(
        base=base,
        weights=weights,
        normalization=normalization,
        metadata=metadata,
        parameter_groups=parameter_groups,
        **kwargs
    )
    return model
