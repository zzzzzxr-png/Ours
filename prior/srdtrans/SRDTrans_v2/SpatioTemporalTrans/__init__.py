import torch
from torch import nn
import complextorch.nn as cvnn
from einops import rearrange
from SRDTrans_v2.SpatioTemporalTrans.TemporalTrans import (
    TemporalTransLayer,
    LearnedPositionalEncoding,
)
from SRDTrans_v2.SpatioTemporalTrans.SpatioiTrans import SpatioTransLayer
from SRDTrans_v2.complex_layers import SharedDropout


class TemporalTransformer(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            input_dropout_rate,
            attn_dropout_rate,
    ):
        super(TemporalTransformer, self).__init__()
        self.transformer = TemporalTransLayer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, seq_length
        )
        self.pre_dropout = SharedDropout(input_dropout_rate)
        self.pre_head_ln = cvnn.LayerNorm(embedding_dim)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=H, p2=W)
        return x


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_layers,
            num_heads,
            window_size,
            hidden_dim,
            attn_drop=0.,
            input_drop=0.,
            post_norm=False,
    ):
        super(SpatioTransformer, self).__init__()
        self.transformer = SpatioTransLayer(
            dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=hidden_dim/embedding_dim,
            qkv_bias=True,
            qk_scale=None,
            drop=input_drop,
            attn_drop=attn_drop,
            drop_path=0.,
            norm_layer=cvnn.LayerNorm,
        )

        self.post_norm = (
            cvnn.LayerNorm(embedding_dim)
            if post_norm
            else nn.Identity()
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b s) (h w) c')
        x = self.transformer(x, H, W)
        x = self.post_norm(x)
        x = rearrange(x, '(b p1) (h p2) c -> b c p1 h p2', p1=D, p2=W)
        return x


class SpatioTemporalTrans(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=2,
            num_space_trans_layer=2,
            trans_order='ts',
            space_post_norm=False,
            space_dropout_rate=0.,
    ):
        super(SpatioTemporalTrans, self).__init__()

        self.trans_order = trans_order.lower()
        if self.trans_order not in ['ts', 'st']:
            raise ValueError(
                "trans_order must be either 'ts' or 'st', "
                f"but got {trans_order}"
            )

        self.timeTrans = TemporalTransformer(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.spaceTrans = SpatioTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_space_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            window_size=space_window_size,
            attn_drop=attn_dropout_rate,
            input_drop=space_dropout_rate,
            post_norm=space_post_norm,
        )

    def forward(self, x):
        if self.trans_order == 'ts':
            x = self.timeTrans(x)
            x = self.spaceTrans(x)
        elif self.trans_order == 'st':
            x = self.spaceTrans(x)
            x = self.timeTrans(x)
        else:
            raise RuntimeError(f"Unexpected trans_order: {self.trans_order}")
        return x
