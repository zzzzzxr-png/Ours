"""3D U-Net denoiser with optional MPGN conditional feature modulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Encoder, Decoder, DoubleConv, create_feature_maps


class MPGNCondition3D(nn.Module):
    """
    Lightweight 3D conditional feature modulation.

    cond_map is a detached MPGN variance/log-variance map with shape:
        [B, 1, T, H, W]

    It is resized to the current feature scale and used to produce
    gamma and beta for residual affine modulation:
        F' = F + gate * (gamma(cond) * F + beta(cond))

    gamma/beta are zero-initialized, so the model is exactly the original
    3D U-Net at initialization. gate is initialized to 1 to keep the
    modulation branch trainable.
    """

    def __init__(self, channels, hidden_channels=8):
        super(MPGNCondition3D, self).__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 2 * channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x, cond_map, return_debug=False):
        if cond_map is None:
            if return_debug:
                return x, {}
            return x

        # The MPGN map is a physical conditioning signal, not a learned branch.
        cond_map = cond_map.detach()

        if cond_map.size()[2:] != x.size()[2:]:
            cond_map = F.interpolate(
                cond_map,
                size=x.size()[2:],
                mode='trilinear',
                align_corners=False,
            )

        gamma_beta = self.to_gamma_beta(cond_map)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        update = self.gate * (gamma * x + beta)
        out = x + update

        if return_debug:
            # Channel-averaged maps for visualization. Shape: [B, 1, T, H, W]
            feature_abs = x.detach().abs().mean(dim=1, keepdim=True)
            update_abs = update.detach().abs().mean(dim=1, keepdim=True)
            relative_effect = update_abs / (feature_abs + 1e-6)
            update_mean = update.detach().mean(dim=1, keepdim=True)

            debug = {
                'cond_resized': cond_map.detach(),
                'gamma_mean': gamma.detach().mean(dim=1, keepdim=True),
                'beta_mean': beta.detach().mean(dim=1, keepdim=True),
                'update_abs': update_abs,
                'update_mean': update_mean,
                'relative_effect': relative_effect,
                'gate': self.gate.detach().view(1),
            }
            return out, debug

        return out


class UNet3D(nn.Module):
    """
    3D U-Net from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        final_sigmoid (bool): if True build a Sigmoid final activation, else Softmax
            (note: the final activation is currently not applied in forward).
        f_maps (int, tuple): feature maps per encoder level; an int expands to a
            geometric progression f_maps * 2^k, k=0..3.
        layer_order (string): order of layers in `SingleConv`, e.g. 'cr' -> Conv3d+ReLU.
        num_groups (int): number of groups for the GroupNorm.
        use_mpgn_cfm (bool): enable MPGN conditional feature modulation in the decoder.
        mpgn_cfm_hidden (int): hidden width of the MPGN modulation MLP.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid,
                 f_maps=64, layer_order='cr', num_groups=8,
                 use_mpgn_cfm=False, mpgn_cfm_hidden=8, **kwargs):
        super(UNet3D, self).__init__()
        self.use_mpgn_cfm = use_mpgn_cfm

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # encoder path: one Encoder per feature-map level
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # decoder path: len(f_maps) - 1 Decoder modules
        decoders = []
        decoder_out_channels = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
            decoder_out_channels.append(out_feature_num)

        self.decoders = nn.ModuleList(decoders)

        if self.use_mpgn_cfm:
            self.mpgn_cfms = nn.ModuleList([
                MPGNCondition3D(ch, hidden_channels=mpgn_cfm_hidden)
                for ch in decoder_out_channels
            ])
        else:
            self.mpgn_cfms = None

        # final 1x1 convolution reduces channels to out_channels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x, cond_map=None, return_debug=False, return_features=False):
        debug_info = {}

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output (it's the 1st in the list)
        encoders_features = encoders_features[1:]

        # decoder part
        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):
            x = decoder(encoder_features, x)
            if self.use_mpgn_cfm and cond_map is not None:
                if return_debug:
                    x, layer_debug = self.mpgn_cfms[i](x, cond_map, return_debug=True)
                    for k, v in layer_debug.items():
                        debug_info[f'decoder_{i}_{k}'] = v
                else:
                    x = self.mpgn_cfms[i](x, cond_map)

        if return_features:
            # Pre-final-conv decoder feature map (P2-S2: heads are applied to a
            # small local crop of this instead of materializing K hypotheses
            # over the full volume via final_conv).
            return x

        x = self.final_conv(x)

        # NOTE: the final activation is intentionally not applied; the network
        # outputs residual/regression values in physical units.

        if return_debug:
            return x, debug_info
        return x


class Network_3D_Unet(nn.Module):
    """Thin wrapper that selects a 3D U-Net generator."""

    def __init__(self, UNet_type='3DUNet', in_channels=1, out_channels=1,
                 f_maps=64, final_sigmoid=True,
                 use_mpgn_cfm=False, mpgn_cfm_hidden=8):
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.use_mpgn_cfm = use_mpgn_cfm

        if UNet_type == '3DUNet':
            self.Generator = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                final_sigmoid=final_sigmoid,
                use_mpgn_cfm=use_mpgn_cfm,
                mpgn_cfm_hidden=mpgn_cfm_hidden,
            )

    def forward(self, x, cond_map=None, return_debug=False, return_features=False):
        return self.Generator(x, cond_map=cond_map, return_debug=return_debug, return_features=return_features)
