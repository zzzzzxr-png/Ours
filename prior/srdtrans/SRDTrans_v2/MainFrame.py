import torch
import torch.nn as nn
from torch.nn import functional as F
import complextorch.nn as cvnn
from torch.utils.checkpoint import checkpoint

from SRDTrans_v2.complex_layers import ComplexRMSNorm3d


class MSConvBeforeTrans(nn.Module):
    """
    Multi-scale local projection before transformer blocks.

    This module is designed to replace only conv_before_trans.
    It does not replace the encoder/decoder convolutions.

    Branches:
        3x3x3:
            preserves the original joint local spatiotemporal bias.
        1x3x3:
            extracts spatial local morphology without mixing time.
        dilated 1x3x3:
            enlarges the spatial receptive field without adding temporal smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super(MSConvBeforeTrans, self).__init__()

        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        self.branch_3d = nn.Sequential(
            cvnn.Conv3d(
                in_channels,
                c1,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            ComplexRMSNorm3d(),
            cvnn.modReLU()
        )

        self.branch_spatial = nn.Sequential(
            cvnn.Conv3d(
                in_channels,
                c2,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1)
            ),
            ComplexRMSNorm3d(),
            cvnn.modReLU()
        )

        self.branch_spatial_dilated = nn.Sequential(
            cvnn.Conv3d(
                in_channels,
                c3,
                kernel_size=(1, 3, 3),
                padding=(0, 2, 2),
                dilation=(1, 2, 2)
            ),
            ComplexRMSNorm3d(),
            cvnn.modReLU()
        )

        self.fuse = nn.Sequential(
            cvnn.Conv3d(out_channels, out_channels, kernel_size=1),
            ComplexRMSNorm3d(),
            cvnn.modReLU()
        )

    def forward(self, x):
        x = torch.cat(
            [
                self.branch_3d(x),
                self.branch_spatial(x),
                self.branch_spatial_dilated(x),
            ],
            dim=1
        )
        return self.fuse(x)


class MainFrame(nn.Module):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1,
            num_layers=0,
            skip_fusion='add',
    ):
        super(MainFrame, self).__init__()
        self.img_dim = img_dim
        self.img_time = img_time
        self.f_maps = f_maps
        self.gradient_checkpointing = True
        self.checkpoint_transformer_only = False
        # 2Conv + Down
        self.encoders = self.temporalSqueeze(
            f_maps=[in_channel] + f_maps
        )

        # up + 2Conv
        self.decoders = self.temporalExcitation(
            f_maps=f_maps[::-1] + [in_channel],
            skip_fusion=skip_fusion,
        )

    def temporalSqueeze(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])

        for idx in range(1, len(f_maps)):
            encoder_layer = SqueezeLayer(
                in_channels=f_maps[idx-1],
                out_channels=f_maps[idx],
            )
            model_list.append(encoder_layer)
        return model_list

    def temporalExcitation(self, f_maps, skip_fusion='add'):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            decoder_layer = ExcitationLayer(
                in_channels=f_maps[idx-1],
                out_channels=f_maps[idx],
                if_up_sample=True,
                skip_fusion=skip_fusion,
            )
            model_list.append(decoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            if self.gradient_checkpointing and not self.checkpoint_transformer_only and self.training and torch.is_grad_enabled():
                before_down, x = checkpoint(encoder, x, use_reentrant=False)
            else:
                before_down, x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, before_down)

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            x = checkpoint(self.process_by_trans, x, use_reentrant=False)
        else:
            x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            if self.gradient_checkpointing and not self.checkpoint_transformer_only and self.training and torch.is_grad_enabled():
                x = checkpoint(decoder, x, encoder_features, use_reentrant=False)
            else:
                x = decoder(x, encoder_features)

        return x


class SqueezeLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(SqueezeLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=True
        )
        self.down_sample = cvnn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        self.down_norm = ComplexRMSNorm3d()

    def forward(self, x):
        before_down = self.conv_net(x)
        x = self.down_norm(self.down_sample(before_down))
        return before_down, x


def _identity_complex_conv(conv):
    with torch.no_grad():
        conv.weight.zero_()
        channels = min(conv.in_channels, conv.out_channels)
        for index in range(channels):
            conv.weight[index, index, 0, 0, 0] = 1 + 0j
        if conv.bias is not None:
            conv.bias.zero_()


class ExcitationLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_up_sample=True,
            kernel_size=3,
            skip_fusion='add',
    ):
        super(ExcitationLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=False
        )
        self.if_up_sample = if_up_sample
        if skip_fusion not in ('add', 'gated_add', 'conv_add'):
            raise ValueError('unknown skip_fusion: {!r}'.format(skip_fusion))
        self.skip_fusion = skip_fusion
        if skip_fusion == 'gated_add':
            self.skip_gate = nn.Parameter(
                torch.ones(in_channels, dtype=torch.cfloat)
            )
        elif skip_fusion == 'conv_add':
            self.skip_projection = cvnn.Conv3d(
                in_channels, in_channels, kernel_size=1, padding=0
            )
            _identity_complex_conv(self.skip_projection.conv)
        self.up_sample = cvnn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(4,3,3), stride=(2,1,1), padding=(1,1,1))
        self.up_norm = ComplexRMSNorm3d()

    def forward(self, x, encoder_features):
        if self.if_up_sample:
            x = self.up_norm(self.up_sample(x))
        if self.skip_fusion == 'gated_add':
            skip = self.skip_gate.view(1, -1, 1, 1, 1) * encoder_features
        elif self.skip_fusion == 'conv_add':
            skip = self.skip_projection(encoder_features)
        else:
            skip = encoder_features
        x = (x + skip) * (2 ** -0.5)
        x = self.conv_net(x)
        return x


class SingleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1
    ):
        super(SingleConv, self).__init__()
        self.add_module('ComplexConv3d',
                        cvnn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        self.add_module('ComplexRMSNorm3d', ComplexRMSNorm3d())
        self.add_module('modReLU', cvnn.modReLU())


class DoubleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_encoder,
            kernel_size=3
    ):
        super(DoubleConv, self).__init__()
        if if_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=1))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=1))
