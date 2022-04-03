from torch import nn


class DenseAttention(nn.Module):  # Like spatial, but for all channels
    """Dense attention module for global attention from GLAMOR; see arXiv paper
    at https://arxiv.org/pdf/2002.02256.pdf
    """

    def __init__(self, planes):
        super(DenseAttention, self).__init__()
        self.dense_conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )
        self.dense_relu1 = nn.LeakyReLU()
        self.dense_conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )
        self.dense_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_conv1(x)
        x = self.dense_relu1(x)
        x = self.dense_conv2(x)
        x = self.dense_sigmoid(x)
        return x
