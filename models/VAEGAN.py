from torch import nn
from torch.nn import functional as F


class VAEGAN(nn.Module):
    def __init__(self, latent_dimensions = 128,):
        self.latent_dimensions = latent_dimensions
        pass
        self.Encoder = self._Encoder(self.latent_dimensions)
        self.Decoder = self._Decoder(self.latent_dimensions)
        self.LatentDiscriminator = self._LatentDiscriminator()
        self.Discriminator = self._Discriminator()

        # self.Encoder.cuda()
        # self.Decoder.cuda()
        # self.LatentDiscriminator.cuda()
        # self.Discriminator.cuda()

        self.Encoder.weights_init()
        self.Decoder.weights_init()
        self.LatentDiscriminator.weights_init()
        self.Discriminator.weights_init()

    def forward(self,x):
        return self.Decoder(self.Encoder(x))

    class _Encoder(nn.Module):
        expansion_base = 64
        def __init__(self, latent_dimensions, init="normal", **kwargs):
            super(_Encoder, self).__init__()
            self.init = init
            self.kwargs = kwargs

            self.conv1 = nn.Conv2d(3, self.expansion_base, 3, 2, 1)
            self.conv2 = nn.Conv2d(self.expansion_base, self.expansion_base*4, 3, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(self.expansion_base*4)
            self.conv3 = nn.Conv2d(self.expansion_base*4, self.expansion_base*8, 3, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(self.expansion_base*8)
            self.conv4 = nn.Conv2d(self.expansion_base*8, latent_dimensions, 3, 1, 0)

            self.lrelu = F.leaky_relu

        def forward(self,x):
            x = self.conv1(x)
            x = self.lrelu(x, 0.2)
            x = self.conv2(x)
            x = self.conv2_bn(x)
            x = self.lrelu(x, 0.2)
            x = self.conv3(x)
            x = self.conv3_bn(x)
            x = self.lrelu(x, 0.2)
            x = self.conv4(x)
            return x        

        def weights_init(self,init="normal"):
            if init != "normal":
                raise NotImplementedError()
            for m in self._modules:
                #m.apply(self.weight_init_normal)
                self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

        def weight_init_normal(self, m, mean=0.0, std=0.02):
            classname = m.__class__.__name__
            if classname.find('ConvTranspose2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Conv2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)



    class _Decoder(nn.Module):
        expansion_base = 64
        def __init__(self, latent_dimensions, init="normal", **kwargs):
            super(_Decoder, self).__init__()
            self.init = init
            self.kwargs = kwargs

            self.dconv1 = nn.ConvTranspose2d(latent_dimensions, expansion_base*4, 3, 1, 0)
            self.dconv1_bn = nn.BatchNorm2d(expansion_base*4)            
            self.dconv2 = nn.ConvTranspose2d(expansion_base*4, expansion_base*4, 3, 2, 1)
            self.dconv2_bn = nn.BatchNorm2d(expansion_base*4)
            self.dconv3 = nn.ConvTranspose2d(expansion_base*4, d, 3, 2, 1)
            self.dconv3_bn = nn.BatchNorm2d(expansion_base)
            self.dconv4 = nn.ConvTranspose2d(expansion_base, 3, 3, 2, 1)

            self.relu = F.relu
            self.tanh = F.tanh

        def forward(self,x):
            x = self.dconv1(x)
            x = self.dconv1_bn(x)
            x = self.relu(x)
            x = self.dconv2(x)
            x = self.dconv2_bn(x)
            x = self.relu(x)
            x = self.dconv3(x)
            x = self.dconv3_bn(x)
            x = self.relu(x)
            x = self.dconv4(x)
            x = self.tanh(x)
            x *= 0.5
            x += 0.5
            return x        

        def weights_init(self,init="normal"):
            if init != "normal":
                raise NotImplementedError()
            for m in self._modules:
                #m.apply(self.weight_init_normal)
                self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

        def weight_init_normal(self, m, mean=0.0, std=0.02):
            classname = m.__class__.__name__
            if classname.find('ConvTranspose2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Conv2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
    
    class _LatentDiscriminator(nn.Module):
        expansion_base = 64
        def __init__(self, latent_dimensions, init="normal", **kwargs):
            super(_LatentDiscriminator, self).__init__()
            self.init = init
            self.kwargs = kwargs

            self.dense1 = nn.Linear(latent_dimensions, self.expansion_base)
            self.dense2 = nn.Linear(self.expansion_base, self.expansion_base*2)
            self.dense3 = nn.Linear(self.expansion_base*2, 1)

            self.lrelu = F.leaky_relu
            self.sigmoid = F.sigmoid

        def forward(self,x):
            x = self.dense1(x)
            x = self.lrelu(x, 0.2)
            x = self.dense2(x)
            x = self.lrelu(x, 0.2)
            x = self.dense3(x)
            x = self.sigmoid(x)
            return x

        def weights_init(self,init="normal"):
            if init != "normal":
                raise NotImplementedError()
            for m in self._modules:
                #m.apply(self.weight_init_normal)
                self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

        def weight_init_normal(self, m, mean=0.0, std=0.02):
            classname = m.__class__.__name__
            if classname.find('ConvTranspose2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Conv2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)


    class _Discriminator(nn.Module):
        expansion_base = 64
        def __init__(self, latent_dimensions, init="normal", **kwargs):
            super(_Discriminator, self).__init__()
            self.init = init
            self.kwargs = kwargs

            self.conv1 = nn.Conv2d(3, self.expansion_base, 3, 2, 1)
            self.conv2 = nn.Conv2d(self.expansion_base, self.expansion_base*4, 3, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(self.expansion_base*4)
            self.conv3 = nn.Conv2d(self.expansion_base*4, self.expansion_base*8, 3, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(self.expansion_base*8)
            self.conv4 = nn.Conv2d(self.expansion_base * 8, 1, 3, 1, 0)

            self.lrelu = F.leaky_relu
            self.sigmoid = F.sigmoid

        def forward(self,x):
            x = self.conv1(x)
            x = self.lrelu(x, 0.2)
            x = self.conv2(x)
            x = self.conv2_bn(x)
            x = self.lrelu(x, 0.2)
            x = self.conv3(x)
            x = self.conv3_bn(x)
            x = self.lrelu(x, 0.2)
            x = self.conv4(x)
            x = self.sigmoid(x)
            return x

        def weights_init(self,init="normal"):
            if init != "normal":
                raise NotImplementedError()
            for m in self._modules:
                #m.apply(self.weight_init_normal)
                self.weight_init_normal(m, self.kwargs.get("mean", 0.0), self.kwargs.get("std", 0.02))

        def weight_init_normal(self, m, mean=0.0, std=0.02):
            classname = m.__class__.__name__
            if classname.find('ConvTranspose2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Conv2d') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, mean, std)
                nn.init.constant_(m.bias, 0.0)

