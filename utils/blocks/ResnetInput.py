from torch import nn
from utils.blocks import InputAttention

class ResnetInput(nn.Module):
    def __init__(self, ia_attention:bool=False) -> None:
        super().__init__()
        
        self.inplanes = 64
        self.ia=ia_attention
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #if norm_layer == "gn":
        #    self.bn1 = nn.GroupNorm2d
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ia_attention = nn.Identity()
        if self.ia:
            self.ia_attention = InputAttention(self.inplanes)   # 64, set above


    def forward(self, x):
        x = self.conv1(x)        
        if self.ia:
            x = self.ia_attention(x) * x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        return x
