from torch import nn


class InputAttention(nn.Module):
    """Input attention module for global/local attention from GLAMOR; see arXiv paper
    at https://arxiv.org/pdf/2002.02256.pdf
    """

    def __init__(self, planes):
        super(InputAttention, self).__init__()
        self.ia_conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.ia_relu1 = nn.LeakyReLU()
        self.ia_conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.ia_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ia_conv1(x)
        x = self.ia_relu1(x)
        x = self.ia_conv2(x)
        x = self.ia_sigmoid(x)
        return x
