from torch import nn
import torch.nn.functional as F
from utils import layers
import torch

class ReidModel(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        super(ReidModel, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.soft_dimensions = soft_dimensions
        self.normalization = normalization if normalization != '' else None
        self.build_base(base, weights, **kwargs)
        
        if self.normalization == 'bn':
            self.feat_norm = nn.BatchNorm1d(self.embedding_dimensions)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "in":
            self.feat_norm = layers.FixedInstanceNorm1d(self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "gn":
            self.feat_norm = nn.GroupNorm(self.embedding_dimensions // 16, self.embedding_dimensions, affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == "ln":
            self.feat_norm = nn.LayerNorm(self.embedding_dimensions,elementwise_affine=True)
            self.feat_norm.bias.requires_grad_(False)
            self.feat_norm.apply(self.weights_init_kaiming)
        elif self.normalization == 'l2':
            self.feat_norm = layers.L2Norm(self.embedding_dimensions,scale=1.0)
        elif self.normalization is None:
            self.feat_norm = None
        else:
            raise NotImplementedError()

        if self.soft_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.soft_dimensions, bias=False)
            self.softmax.apply(self.weights_init_softmax)
        else:
            self.softmax = None

    def forward(self,x):
        features = self.base_forward(x)
        
        if self.feat_norm is not None:
            inference = self.feat_norm(features)
        else:
            inference = features

        if self.training:
            if self.softmax is not None:
                soft_logits = self.softmax(inference)
            else:
                soft_logits = None
            return soft_logits, features
        else:
            return inference

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
    
    def build_base(self,):
        raise NotImplementedError()
    def base_forward(self,x):
        raise NotImplementedError()

    def partial_load(self,weights_path):
        params = torch.load(weights_path)
        for _key in params:
            if _key not in self.state_dict().keys() or params[_key].shape != self.state_dict()[_key].shape: 
                continue
            self.state_dict()[_key].copy_(params[_key])

    class LambdaLayer(nn.Module):
        """ Torch lambda layer to act as an empty layer. It does not do anything """
        def __init__(self, lambd):
                super(LambdaLayer, self).__init__()
                self._lambda = lambd
        def forward(self, x):
                return self._lambda(x)

def veri_model_builder(arch, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
    # First identify the architecture...
    arch = arch+"Base"
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)

    model = archbase(base = base, weights=weights, normalization = normalization, embedding_dimensions = embedding_dimensions, soft_dimensions = soft_dimensions, **kwargs)
    return model

   
    
def vaegan_model_builder(arch, base, latent_dimensions = None, **kwargs):
    archbase = __import__("models."+arch, fromlist=[arch])
    archbase = getattr(archbase, arch)
    
    return archbase(base, latent_dimensions=latent_dimensions, **kwargs)
    

model_builder = veri_model_builder


