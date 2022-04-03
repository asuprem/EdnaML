import torch
from torch import nn
import torch.nn.init
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class LambdaLayer(nn.Module):
    """ Torch lambda layer to act as an empty layer. It does not do anything """

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self._lambda = lambd

    def forward(self, x):
        return self._lambda(x)


# https://github.com/amdegroot/ssd.pytorch/blob/master/layers/modules/l2norm.py
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.expand_as(x) * x
        return out


from torch.nn.modules import instancenorm

# https://github.com/pytorch/pytorch/pull/9924/commits/816d048b91a455a5f57da6f1ab304e70c38a39bb
class FixedInstanceNorm1d(instancenorm._InstanceNorm):
    """Applies Instance Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization`_ .

    TODO Fix this formatting to match sphinx documentation

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.
    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    Examples::
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = torch.randn(20, 100, 40)
        >>> output = m(input)
    .. _`Instance Normalization: The Missing Ingredient for Fast Stylization`:
        https://arxiv.org/abs/1607.08022
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


def group_norm(
    input,
    group,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError(
            "Expected running_mean and running_var to be not None when use_input_stats=False"
        )

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(
        input,
        group,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        use_input_stats=None,
        momentum=None,
        eps=None,
    ):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        # norm_shape = [1, b * c / group, group]
        # print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(
            1, int(b * c / group), group, *input.size()[2:]
        )

        out = F.batch_norm(
            input_reshaped,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=use_input_stats,
            momentum=momentum,
            eps=eps,
        )

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(
                running_mean.view(b, int(c / group)).mean(0, keepdim=False)
            )
        if running_var is not None:
            running_var_orig.copy_(
                running_var.view(b, int(c / group)).mean(0, keepdim=False)
            )

        return out.view(b, c, *input.size()[2:])

    return _instance_norm(
        input,
        group,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )


class _GroupNorm(_BatchNorm):
    def __init__(
        self,
        num_features,
        num_groups=1,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(
            int(num_features / num_groups), eps, momentum, affine, track_running_stats
        )

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input,
            self.num_groups,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class GroupNorm2d(_GroupNorm):
    r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    https://arxiv.org/pdf/1803.08494.pdf
    `Group Normalization`_ .
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        num_groups:
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class GroupNorm3d(_GroupNorm):
    """
        Assume the data format is (B, C, D, H, W)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
