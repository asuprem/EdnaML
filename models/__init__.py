from torch import nn

class ReidModel(nn.Module):
    def __init__(self, base, weights=None, normalization=None, embedding_dimensions=None, soft_dimensions=None, **kwargs):
        super(ReidModel, self).__init__()
        self.base = None
        
        self.embedding_dimensions = embedding_dimensions
        self.soft_dimensions = soft_dimensions
        self.normalization = normalization

        self.build_base(base, weights, **kwargs)

        
        if self.normalization == 'bn':
            self.bn_bottleneck = nn.BatchNorm1d(self.embedding_dimensions)
            self.bn_bottleneck.bias.requires_grad_(False)
            self.bn_bottleneck.apply(self.weights_init_kaiming)
        elif self.normalization is None:
            self.bn_bottleneck = None

        if self.soft_dimensions is not None:
            self.softmax = nn.Linear(self.embedding_dimensions, self.soft_dimensions, bias=False)
            self.softmax.apply(self.weights_init_softmax)
        else:
            self.softmax = None

    def forward(self,x):
        features = self.base_forward(x)
        
        if self.bn_bottleneck is not None:
            inference = self.bn_bottleneck(features)
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

    class LambdaLayer(nn.Module):
        """ Torch lambda layer to act as an empty layer. It does not do anything """
        def __init__(self, lambd):
                super(LambdaLayer, self).__init__()
                self._lambda = lambd
        def forward(self, x):
                return self._lambda(x)
