from torch import nn
import torch.nn.functional as F
import torch

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

    
class ReidModel(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        """Basic ReID Resnet model.

        A ReID model performs re-identification by generating embeddings such that the same class's embeddings are closer together.

        Args:
            base (str): The architecture base for resnet, i.e. resnet50, resnet18
            weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
            normalization (str, None): Can be None, where no normalization is used. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
            embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.
            soft_dimensions (int, None): Whether to include softmax classification layer. If None, softmax layer is not created in model.
        
        Kwargs (MODEL_KWARGS):
            last_stride (int, 1): The final stride parameter for the architecture core. Should be one of 1 or 2.
            attention (str, None): The attention module to use. Only supports ['cbam', None]
            input_attention (bool, false): Whether to include the IA module
            ia_attention (bool, false): Whether to include input IA module
            part_attention (bool, false): Whether to include Part-CBAM Mobule
            secondary_attention (int, None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic block number provided here.

        Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS; set in backbones.resnet):
            zero_init_residual (bool, false): Whether the final layer uses zero initialization
            top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
            num_classes (int, 1000): Number of features in final imagenet FC layer
            groups (int, 1): Used during resnet variants construction
            width_per_group (int, 64): Used during resnet variants construction
            replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
            norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D

        """
        super(ReidModel, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.soft_dimensions = soft_dimensions
        self.normalization = normalization if normalization != '' else None
        self.build_base(base, weights, **kwargs)    # All kwargs are passed into build_base,, which in turn passes kwargs into _resnet()
        
        self.feat_norm = None
        self.build_normalization(self.normalization)
        
        if self.soft_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.soft_dimensions, bias=False)
            self.softmax.apply(self.weights_init_softmax)
        else:
            self.softmax = None

    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('GroupNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('LayerNorm') != -1:
                #if m.affine:
                #    nn.init.constant_(m.weight, 1.0)
                #    nn.init.constant_(m.bias, 0.0)
                pass
        elif classname.find('InstanceNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def weights_init_softmax(self, m):
        """ Initialize linear weights to standard normal. Mean 0. Standard Deviation 0.001 """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, std=0.001)
                if m.bias:
                        nn.init.constant_(m.bias, 0.0)
    
    def partial_load(self,weights_path):
        params = torch.load(weights_path)
        for _key in params:
            if _key not in self.state_dict().keys() or params[_key].shape != self.state_dict()[_key].shape: 
                continue
            self.state_dict()[_key].copy_(params[_key])


    def build_base(self,**kwargs):
        """Build the architecture base.        
        """
        raise NotImplementedError()
    def build_normalization(self,**kwargs):
        raise NotImplementedError()
    def base_forward(self,**kwargs):
        raise NotImplementedError()
    def forward(self,**kwargs):
        raise NotImplementedError()
