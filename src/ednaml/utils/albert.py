import torch
from torch import nn


def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    import os
    import torch

    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be"
            " installed. Please see https://www.tensorflow.org/install/ for"
            " installation instructions."
        )
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    if not os.path.exists(tf_path + "/checkpoint"):
        tf_path = tf_path + "/variables/variables"
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.replace("attention_1", "attention")
        name = name.replace("ffn_1", "ffn")
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            elif re.fullmatch(r"[A-Za-z]+_+[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] in ["LayerNorm", "attention", "ffn"] and len(l) >= 2:
                l = ["_".join(l[:-1])]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif l[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def prune_linear_layer(layer, index, dim=0):
    """Prune a linear layer (a model parameters) to keep only entries in index.
    Return the pruned layer as a new layer with requires_grad=True.
    Used to remove heads.
    """
    from torch import nn

    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_conv1d_layer(layer, index, dim=1):
    """Prune a Conv1D layer (a model parameters) to keep only entries in index.
    A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
    Return the pruned layer as a new layer with requires_grad=True.
    Used to remove heads.
    """
    from ednaml.utils.layers import Conv1D

    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer, index, dim=None):
    """Prune a Conv1D or nn.Linear layer (a model parameters) to keep only entries in index.
    Return the pruned layer as a new layer with requires_grad=True.
    Used to remove heads.
    """
    from ednaml.utils.layers import Conv1D
    from torch import nn

    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError("Can't prune layer of class {}".format(layer.__class__))


class AlbertEmbeddingAverage(nn.Module):
    """Averages embeddings on given dimension

    Args:
        nn (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs):
        return torch.mean(outputs[0], dim=1)


class AlbertPooledOutput(nn.Module):
    """Torch layer to return a single index from list. It does not do anything"""

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        return outputs[1]


class AlbertRawCLSOutput(nn.Module):
    """Torch layer to return a single index from list. It does not do anything"""

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        return outputs[0][:, 0]
