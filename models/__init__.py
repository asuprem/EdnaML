from multiprocessing.sharedctypes import Value


def veri_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, softmax_dimensions=None, **kwargs):
    """Vehicle Re-id model builder.

    This builds a model for vehicle re-id. Refer to paper [] for general construction. The model contains:
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
        softmax_dimensions (int): Feature dimensions for softmax FC layer. Can be None if not using it. Should be equal to the number of identities in the dataset.
        kwargs (dict): Nothing yet


    Returns:
        Torch Model: A Torch Re-ID model

    """
    arch = arch+"Base"
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, softmax_dimensions = softmax_dimensions, **kwargs)
    return model

def vaegan_model_builder(arch, base, latent_dimensions = None, **kwargs):
    """VAE-GAN Model builder.

    This builds the VAAE-GAN model used in paper []

    Args:
        arch (str): The architecture to use. Only "VAEGAN" is supported.
        base (int): Dimensionality of the input images. MUST be a power of 2.
        latent_dimensions (int): Embedding dimension for the encoder of the VAAE-GAN.
        
        kwargs (dict): Needs the following:
            channels (int): Number of channels in image. Default 3. 1 for MNIST.

    Returns:
        Torch Model: A Torch-based VAAE-GAN model
    """
    if arch != "VAEGAN":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)
    
    return archbase(base, latent_dimensions=latent_dimensions, **kwargs)
    

def carzam_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, **kwargs):
    """CarZam model builder.

    This builds a simple single model (NOT teamed classifier) for CarZam. Refer to paper [] for general construction. The model contains:
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
        softmax_dimensions (int): Feature dimensions for softmax FC layer. Can be None if not using it. Should be equal to the number of identities in the dataset.
        kwargs (dict): Nothing yet


    Returns:
        Torch Model: A Torch Re-ID model

    """
    if arch != "CarzamResnet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, **kwargs)
    return model


def classification_model_builder(arch, base, weights=None, normalization=None, **kwargs):
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
    softdim = kwargs.get("metadata")
    if type(softdim) is not int:
        print("Softmax dimensions not provided as int. Attempting to infer number of classes")
        if type(softdim) is dict:
            if len(softdim)>1:
                raise ValueError("More than one annotation provided. Use a multiclassification model or multibranch model instead")
            key=softdim.keys()[0]
            softdim=softdim[key]
        else:
            raise RuntimeError("Softmax dimensions not provided as int or dictionary")
    kwargs["softmax_dimensions"] = softdim

    model = archbase(base = base, weights=weights, normalization = normalization, **kwargs)
    return model



def multiclassification_builder(arch, base, weights=None, normalization=None, **kwargs):
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

    Kwargs:
        number_outputs (int): Number of different FC layers connected to the feature layer of the backbone.
        softmax_dimensions (List[int]): Classes of each FC layer, in order.
        output_classnames (List[str]): The name for each output, in order. These should be the same as the label names for the multiple classes.
        labelnames (List[str]): The order of labels provided by the crawler. This is used during model training, where crawler ground truth labels must be matched to model outputs. 

    Returns:
        Torch Model: A Torch Re-ID model

    """

    # make the MultiClassificationResNet, with resnet base, with multiple output fc layers
    if arch != "MultiClassificationResNet":
        raise NotImplementedError()
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    
    # Verify that softmax_dimensions is dictionary of annotation->numclasses. 
    softdim = kwargs.get("softmax_dimensions")
    if type(softdim) is not dict:
        raise ValueError("Did not provide dictionary of labels-to-numclasses to built multiclassification FC-layer. If there is only one FC, a singleton dict must be provided")

    model = archbase(base = base, weights=weights, normalization = normalization, **kwargs)
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