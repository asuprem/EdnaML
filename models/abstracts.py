from torch import nn
import torch.nn.functional as F
import torch


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

    



class ClassificationResnetAbstract(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, softmax_dimensions=None, **kwargs):
        """Basic Classification Resnet model.

        A CoLabel model performs classification using corroboratively labeled data to generate labels for unlabeed data.

        Args:
            base (str): The architecture base for resnet, i.e. resnet50, resnet18
            weights (str, None): Path to weights file for the architecture base ONLY. If not provided, base initialized with random values.
            normalization (str, None): Can be None, where no normalization is used. Else create a normalization layer. Supports: ["bn", "l2", "in", "gn", "ln"]
            embedding_dimensions (int): Dimensions for the feature embedding. Leave empty if feature dimensions should be same as architecture core output (e.g. resnet50 base model has 2048-dim feature outputs). If providing a value, it should be less than the architecture core's base feature dimensions.
            softmax_dimensions (int, None): Number of FC dimensions for classification
        
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

        """
        super(ClassificationResnetAbstract, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.softmax_dimensions = softmax_dimensions
        self.normalization = normalization if normalization != '' else None
        self.build_base(base, weights, **kwargs)    # All kwargs are passed into build_base,, which in turn passes kwargs into _resnet()
        
        self.feat_norm = None
        self.build_normalization(self.normalization)
        
        if self.softmax_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.softmax_dimensions, bias=False)
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



# prework notes -- copying over code to this format -- fix the inconsistencies

class CoLabelInterpretableResnetAbstract(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        """Basic CoLabel Interpretable Resnet model.

        A CoLabel model performs classification using corroboratively labeled data to generate labels for unlabeed data.
        
        assuming branches=3; we generate a resnet model with 3 branches, each with their feature output.

        These feature outputs are fed to a softmax, plus concatenated for final outputs.


        kwargs.branches determines how many complementary feature branches are in this colabel model

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
            secondary_attention (List[int], None): Whether to modify CBAM to apply it to specific Resnet basic blocks. None means CBAM is applied to all. Otherwise, CBAM is applied only to the basic block number provided here.

        Default Kwargs (DO NOT CHANGE OR ADD TO MODEL_KWARGS; set in backbones.resnet):
            zero_init_residual (bool, false): Whether the final layer uses zero initialization
            top_only (bool, true): Whether to keep only the architecture base without imagenet fully-connected layers (1000 classes)
            num_classes (int, 1000): Number of features in final imagenet FC layer
            groups (int, 1): Used during resnet variants construction
            width_per_group (int, 64): Used during resnet variants construction
            replace_stride_with_dilation (bool, None): Well, replace stride with dilation...
            norm_layer (nn.Module, None): The normalization layer within resnet. Internally defaults to nn.BatchNorm2D

        """
        super(CoLabelInterpretableResnetAbstract, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.soft_dimensions = soft_dimensions
        self.normalization = normalization if normalization != '' else None
        #This is the feature extractor
        self.build_base(base, weights, **kwargs)    # All kwargs are passed into build_base,, which in turn passes kwargs into _resnet()
        
        #The specific type of ending feature normalizer
        self.feat_norm = None
        self.build_normalization(self.normalization)
        
        # This is the softmax dimensions...
        self.build_softmax()

    def build_softmax(self):
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