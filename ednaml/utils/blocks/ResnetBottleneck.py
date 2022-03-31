from torch import nn

from ednaml.utils.blocks import ChannelAttention
from ednaml.utils.blocks import DenseAttention
from ednaml.utils.blocks import InputAttention
from ednaml.utils.blocks import SpatialAttention

class ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(   self, inplanes, planes, stride=1, downsample=None, groups = 1, base_width = 64, dilation = 1, norm_layer=None, 
                    attention = None, input_attention=False, part_attention=False):
        super(ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,stride=1)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        if input_attention:
            self.input_attention = InputAttention(planes)
        else:
            self.input_attention = None

        if attention is None:
            self.ca = None
            self.sa = None
        elif attention == 'cbam':
            self.sa = SpatialAttention(kernel_size=3)
            self.ca = ChannelAttention(planes*self.expansion)
        elif attention == 'dbam':
            self.ca = ChannelAttention(planes)
            self.sa = DenseAttention(planes)
        else:
            raise NotImplementedError()

        if part_attention:
            self.p_sa = DenseAttention(planes=planes*self.expansion)
            self.p_ca = ChannelAttention(planes*self.expansion)
        else:
            self.p_ca = None
            self.p_sa = None
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.input_attention is not None:
            x = self.input_attention(x) * x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.ca is not None:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        p_out = out
        part_mask = None
        if self.p_ca is not None:   # Get part attention
            p_out = self.p_sa(p_out) * p_out
#            p_out = self.p_ca(p_out) * p_out
            p_out = self.relu(p_out)
            part_mask = self.p_ca(p_out)
        
        out = out + identity
        out = self.relu(out)

        if self.p_ca is not None:   # Concat part attention
            #out = torch.cat([p_out[:,p_out.shape[1]//2:,:,:],out[:,:p_out.shape[1]//2,:,:]],dim=1)
            out = (part_mask * p_out) + ((1-part_mask)*out)
        return out