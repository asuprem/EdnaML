import torch
from torch import nn
from ednaml.utils.blocks import ChannelAttention, DenseAttention, InputAttention


class SELayer(nn.Module):
    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            # if the input is (N, C, H, W)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
            )
        else:
            # if the input is (N, C)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(inplanes, inplanes // 4, bias=False),
                nn.BatchNorm1d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // 4, inplanes, bias=False),
            )

    def forward(self, x):
        atten = self.SE_opr(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten


class HS(nn.Module):
    def __init__(self):
        super(HS, self).__init__()

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip


class Shufflenet(nn.Module):
    def __init__(
        self, inp, oup, base_mid_channels, *, ksize, stride, activation, useSE
    ):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup // 2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = nn.ModuleList(
            [
                # pw
                nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                None,
                # dw
                nn.Conv2d(
                    base_mid_channels,
                    base_mid_channels,
                    ksize,
                    stride,
                    pad,
                    groups=base_mid_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(base_mid_channels),
                # pw-linear
                nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                None,
            ]
        )
        if activation == "ReLU":
            assert useSE == False
            """This model should not have SE with ReLU"""
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = nn.ModuleList(
                [
                    # dw
                    nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    # pw-linear
                    nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(inp),
                    None,
                ]
            )
            if activation == "ReLU":
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):
    def __init__(self, inp, oup, base_mid_channels, *, stride, activation, useSE):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup // 2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = nn.ModuleList(
            [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw
                nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                None,
                # dw
                nn.Conv2d(
                    base_mid_channels,
                    base_mid_channels,
                    3,
                    stride,
                    1,
                    groups=base_mid_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(base_mid_channels),
                # pw
                nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(base_mid_channels),
                None,
                # dw
                nn.Conv2d(
                    base_mid_channels,
                    base_mid_channels,
                    3,
                    stride,
                    1,
                    groups=base_mid_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(base_mid_channels),
                # pw
                nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputs),
                None,
            ]
        )

        if activation == "ReLU":
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != "ReLU"
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = nn.ModuleList(
                [
                    # dw
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    # pw-linear
                    nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(inp),
                    None,
                ]
            )
            if activation == "ReLU":
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % 4 == 0
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2_Plus(nn.Module):
    def __init__(
        self,
        input_size=224,
        architecture=None,
        model_size="Large",
        ia_attention=True,
        part_attention=True,
    ):
        super(ShuffleNetV2_Plus, self).__init__()

        assert input_size % 32 == 0
        assert architecture is not None

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == "Large":
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == "Medium":
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == "Small":
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        )

        self.features = nn.ModuleList([])
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            activation = "HS" if idxstage >= 1 else "ReLU"
            useSE = "True" if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    # print('Shuffle3x3')
                    self.features.append(
                        Shufflenet(
                            inp,
                            outp,
                            base_mid_channels=outp // 2,
                            ksize=3,
                            stride=stride,
                            activation=activation,
                            useSE=useSE,
                        )
                    )
                elif blockIndex == 1:
                    # print('Shuffle5x5')
                    self.features.append(
                        Shufflenet(
                            inp,
                            outp,
                            base_mid_channels=outp // 2,
                            ksize=5,
                            stride=stride,
                            activation=activation,
                            useSE=useSE,
                        )
                    )
                elif blockIndex == 2:
                    # print('Shuffle7x7')
                    self.features.append(
                        Shufflenet(
                            inp,
                            outp,
                            base_mid_channels=outp // 2,
                            ksize=7,
                            stride=stride,
                            activation=activation,
                            useSE=useSE,
                        )
                    )
                elif blockIndex == 3:
                    # print('Xception')
                    self.features.append(
                        Shuffle_Xception(
                            inp,
                            outp,
                            base_mid_channels=outp // 2,
                            stride=stride,
                            activation=activation,
                            useSE=useSE,
                        )
                    )
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        # self.features = nn.Sequential(*self.features)      # manually do it for attention....

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            HS(),
        )
        self.globalpool = nn.AvgPool2d(7)

        """ Don't need these.
        self.LastSE = SELayer(1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 1280, bias=False),
            HS(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(1280, n_class, bias=False))
        """
        # Create true IA
        if ia_attention:
            self.ia_attention = InputAttention(
                16
            )  # MAGIC...use from architecture array
        else:
            self.ia_attention = None

        if part_attention:
            self.p_sa = DenseAttention(36)  # MAGIC...use from architecture array
            self.p_ca = ChannelAttention(36)
            self.p_relu = nn.ReLU(inplace=True)
        else:
            self.p_ca = None
            self.p_sa = None
            self.p_relu = None

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        if self.ia_attention is not None:
            x = self.ia_attention(x) * x

        x = self.features[0](x)
        if self.p_ca is not None:
            p_out = self.p_sa(x) * x
            p_out = self.p_relu(p_out)
            part_mask = self.p_ca(p_out)
            x = (part_mask * p_out) + ((1 - part_mask) * x)

        for idx, layer in enumerate(self.features[1:]):
            x = layer(x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        """x = self.LastSE(x)

        x = x.contiguous().view(-1, 1280)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)"""
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name or "SE" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_param(self, weights_path):
        params = torch.load(weights_path)
        if "state_dict" in params:
            # This is shufflenet style, not ours...
            for i in params["state_dict"]:  # ShuffleNet weights specific issue
                if "fc" in i:
                    continue
                # All 'i' in params have "module." in front of their layer names. [7:] gets rid of it for our models...
                if (
                    i[7:] not in self.state_dict()
                    or params["state_dict"][i].shape != self.state_dict()[i[7:]].shape
                ):
                    continue
                self.state_dict()[i[7:]].copy_(params["state_dict"][i])
        else:  # Our style
            for _key in params:
                if (
                    _key not in self.state_dict().keys()
                    or params[_key].shape != self.state_dict()[_key].shape
                ):
                    continue
                self.state_dict()[_key].copy_(params[_key])


def _shufflenetv2_plus(model_size="Small", **kwargs):
    # Set up shufflenet architecture (from https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2%2B/train.py)
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(
        architecture=architecture, model_size=model_size, **kwargs
    )
    return model


def shufflenetv2_small(**kwargs):
    """ShuffleNetv2-Small model from https://github.com/megvii-model/ShuffleNet-Series/
    """
    return _shufflenetv2_plus(model_size="Small", **kwargs)
