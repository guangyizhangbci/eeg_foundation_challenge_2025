import copy
from typing import Optional, Callable, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 norm_first: bool = False, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        half = d_model // 2
        half_heads = nhead // 2
        self.self_attn_s = nn.MultiheadAttention(half, half_heads, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(half, half_heads, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.linear1   = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2   = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm1     = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2     = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout   = nn.Dropout(dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.norm_first = norm_first

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        bz, ch, patches, patch_size = x.shape
        half = patch_size // 2

        xs = x[:, :, :, :half].transpose(1, 2).contiguous().view(bz * patches, ch, half)
        xt = x[:, :, :, half:].contiguous().view(bz * ch, patches, half)

        xs = self.self_attn_s(xs, xs, xs, attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask, need_weights=False)[0]
        xt = self.self_attn_t(xt, xt, xt, attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask, need_weights=False)[0]

        xs = xs.contiguous().view(bz, patches, ch, half).transpose(1, 2)
        xt = xt.contiguous().view(bz, ch, patches, half)
        return self.dropout1(torch.cat((xs, xt), dim=3))

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":  return F.relu
    if activation == "gelu":  return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


if __name__ == '__main__':
    layer   = TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024,
                                      batch_first=True, norm_first=True, activation=F.gelu)
    encoder = TransformerEncoder(layer, num_layers=2).cuda()
    a = torch.randn(4, 19, 30, 256).cuda()
    print(encoder(a).shape)
