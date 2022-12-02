from torch import nn

from ednaml.utils.blocks import ChannelAttention
from ednaml.utils.blocks import DenseAttention
from ednaml.utils.blocks import InputAttention
from ednaml.utils.blocks import SpatialAttention


class ResnetBasicBlock(nn.Module):
    """ResNet component block for R18, R34. Useful when there are not too many layers
    to deal with complicated matrix multiplications.

    Raises:
        ValueError: When groups!=1 or base_width!=64
        NotImplementedError: When dilation>1 or attention not cbam or dbam
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=nn.BatchNorm2d,
        attention: str = None,
        input_attention: bool = False,
        part_attention: bool = False,
    ):
        """Sets up the ResNet BasicBlock

        Args:
            inplanes (int): Depth of input
            planes (int): Depth of output
            stride (int, optional): Convolutional stride parameter. Defaults to 1.
            downsample (Union[int,None], optional): Whether to downsample images for skip connection. Defaults to None.
            groups (int, optional): <>. Defaults to 1.
            base_width (int, optional): <>. Defaults to 64.
            dilation (int, optional): Convolutional dilation parameters. Defaults to 1.
            norm_layer (Union[nn.GroupNorm,nn.modules.batchnorm._NormBase,None], optional): Normalization layer throughout Block. Defaults to nn.BatchNorm2d.
            attention (str, optional): Attention type: CBAM or DBAM. Defaults to None.
            input_attention (bool, optional): Whether to include the input attention module. Defaults to False.
            part_attention (bool, optional): Whether to include the part (local) attention module. Defaults to False.
        """
        super(ResnetBasicBlock, self).__init__()
        # Verify some base parameters
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if attention is not None and attention not in ["cbam", "dbam"]:
            raise ValueError(
                "attention parameter is unsupported value %s. Use one of"
                " 'cbam','dbam'." % attention
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # This builds the core layers in the BasicBlock
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=1,
            dilation=1,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
            dilation=1,
        )
        self.bn2 = norm_layer(planes)

        # Sets up input attention
        self.input_attention = None
        if input_attention:
            self.input_attention = InputAttention(planes)

        # Sets up CBAM or DBAM attention
        self.ca = None
        self.sa = None
        if attention == "cbam":
            self.ca = ChannelAttention(planes)
            self.sa = SpatialAttention(kernel_size=3)
        if attention == "dbam":
            self.ca = ChannelAttention(planes)
            self.sa = DenseAttention(planes)

        # Sets up local attention
        self.p_ca = None
        self.p_sa = None
        if part_attention:
            self.p_sa = DenseAttention(planes=planes * self.expansion)
            self.p_ca = ChannelAttention(planes * self.expansion)

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

        if self.ca is not None:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        p_out = out
        part_mask = None
        if self.p_ca is not None:  # Get part attention
            p_out = self.p_sa(p_out) * p_out
            #            p_out = self.p_ca(p_out) * p_out
            p_out = self.relu(p_out)
            part_mask = self.p_ca(p_out)

        out = out + identity
        out = self.relu(out)

        if self.p_ca is not None:  # Concat part attention
            # out = torch.cat([p_out[:,p_out.shape[1]//2:,:,:],out[:,:p_out.shape[1]//2,:,:]],dim=1)
            out = (part_mask * p_out) + ((1 - part_mask) * out)
        return out
