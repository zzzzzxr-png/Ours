import torch
from torch import nn
import complextorch.nn as cvnn
from einops import rearrange
from SRDTrans_v2.SpatioTemporalTrans.TemporalTrans import (
    TemporalTransLayer,
    LearnedPositionalEncoding,
)
from SRDTrans_v2.SpatioTemporalTrans.SpatioiTrans import SpatioTransLayer
from SRDTrans_v2.SpatioTemporalTrans.ComplexRestormer import (
    ComplexRestormerBlock,
    ComplexVideoChannelNorm,
)
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
            add_position=True,
            final_norm=True,
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
        self.add_position = bool(add_position)
        self.position_encoding = (
            LearnedPositionalEncoding(embedding_dim, seq_length)
            if self.add_position else None
        )
        self.pre_dropout = SharedDropout(input_dropout_rate)
        self.pre_head_ln = cvnn.LayerNorm(embedding_dim) if final_norm else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        if self.add_position:
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
            shift_size=None,
            attention_type='swin',
        ):
        super(SpatioTransformer, self).__init__()
        self.attention_type = str(attention_type).lower()
        if self.attention_type == 'swin':
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
                shift_size=shift_size,
            )
        elif self.attention_type == 'restormer':
            self.transformer = nn.ModuleList([
                ComplexRestormerBlock(
                    embedding_dim, num_heads, hidden_dim, dropout=input_drop
                ) for _ in range(num_layers)
            ])
        else:
            raise ValueError("attention_type must be 'swin' or 'restormer'")

        if post_norm and self.attention_type == 'restormer':
            self.post_norm = ComplexVideoChannelNorm(embedding_dim)
        elif post_norm:
            self.post_norm = cvnn.LayerNorm(embedding_dim)
        else:
            self.post_norm = nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.size()
        if self.attention_type == 'swin':
            x = rearrange(x, 'b c s h w -> (b s) (h w) c')
            x = self.transformer(x, H, W)
            x = rearrange(x, '(b p1) (h p2) c -> b c p1 h p2', p1=D, p2=W)
        else:
            x = rearrange(x, 'b c s h w -> (b s) c h w')
            for block in self.transformer:
                x = block(x)
            x = rearrange(x, '(b p1) c h w -> b c p1 h w', p1=D)
        x = self.post_norm(x)
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
            interleaved=False,
            attention_type='swin',
    ):
        super(SpatioTemporalTrans, self).__init__()

        self.trans_order = trans_order.lower()
        if self.trans_order not in ['ts', 'st']:
            raise ValueError(
                "trans_order must be either 'ts' or 'st', "
                f"but got {trans_order}"
            )

        self.interleaved = bool(interleaved)
        if self.interleaved:
            self.space_regular = SpatioTransformer(
                embedding_dim=embedding_dim,
                num_layers=1,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                window_size=space_window_size,
                attn_drop=attn_dropout_rate,
                input_drop=space_dropout_rate,
                post_norm=False,
                shift_size=0,
                attention_type=attention_type,
            )
            self.time_first = TemporalTransformer(
                seq_length=seq_length,
                embedding_dim=embedding_dim,
                num_layers=1,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                input_dropout_rate=input_dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                add_position=True,
                final_norm=False,
            )
            self.space_shifted = SpatioTransformer(
                embedding_dim=embedding_dim,
                num_layers=1,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                window_size=space_window_size,
                attn_drop=attn_dropout_rate,
                input_drop=space_dropout_rate,
                post_norm=space_post_norm,
                shift_size=space_window_size // 2,
                attention_type=attention_type,
            )
            self.time_second = TemporalTransformer(
                seq_length=seq_length,
                embedding_dim=embedding_dim,
                num_layers=1,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                input_dropout_rate=input_dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                add_position=False,
                final_norm=True,
            )
            return

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
            attention_type=attention_type,
        )

    def forward(self, x):
        if self.interleaved:
            x = self.space_regular(x)
            x = self.time_first(x)
            x = self.space_shifted(x)
            return self.time_second(x)
        if self.trans_order == 'ts':
            x = self.timeTrans(x)
            x = self.spaceTrans(x)
        elif self.trans_order == 'st':
            x = self.spaceTrans(x)
            x = self.timeTrans(x)
        else:
            raise RuntimeError(f"Unexpected trans_order: {self.trans_order}")
        return x
