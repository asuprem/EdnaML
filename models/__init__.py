from utils.LabelMetadata import LabelMetadata



def classification_model_builder(arch, base, weights=None, normalization=None, metadata:LabelMetadata = None, **kwargs):
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
        Torch Model: A Torch Re-ID model

    """
    if arch != "ClassificationResnet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)


    # Extract the dimensions from the classes metadata provided by EdnaML
    kwargs["softmax_dimensions"] = metadata.getLabelDimensions()

    model = archbase(base = base, weights=weights, normalization = normalization, metadata = metadata, **kwargs)
    return model



def multiclassification_model_builder(arch, base, weights=None, normalization=None, metadata=None, **kwargs):
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
        number_outputs (int): Number of different FC layers connected to the feature layer of the backbone.
        softmax_dimensions (List[int]): Classes of each FC layer, in order.
        output_classnames (List[str]): The name for each output, in order. These should be the same as the label names for the multiple classes.
        labelnames (List[str]): The order of labels provided by the crawler. This is used during model training, where crawler ground truth labels must be matched to model outputs. 

    Returns:
        Torch Model: A Torch Re-ID model

    """

    # make the MultiClassificationResnet, with resnet base, with multiple output fc layers
    if arch != "MultiClassificationResnet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, metadata=metadata, **kwargs)
    return model


def colabel_interpretable_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, **kwargs):
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
        embedding_dimension (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions
        soft_dimensions (int): Feature dimensions for softmax FC layer. Can be None if not using it. Should be equal to the number of identities in the dataset.
        kwargs (dict): branches: 3,4,5 etc


    Returns:
        Torch Model: A Torch Re-ID model

    """
    if arch != "CoLabelInterpretableResnet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, **kwargs)
    return model