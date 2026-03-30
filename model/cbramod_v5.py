import torch
import torch.nn as nn
import torch.nn.functional as F

from criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder

SELECTED_CHANNELS = [4, 5, 6, 12, 30, 34, 36, 40, 41, 51, 52, 53, 54, 60,
                     78, 79, 85, 86, 91, 92, 97, 102, 104, 105, 109, 110, 111, 116, 117]


class CBraMod(nn.Module):
    def __init__(self, in_dim=100, out_dim=100, d_model=100, dim_feedforward=800,
                 seq_len=8, n_layer=24, nhead=10, param=None):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, d_model, param)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                    dim_feedforward=dim_feedforward,
                                    batch_first=True, norm_first=True,
                                    activation=F.gelu),
            num_layers=n_layer,
        )
        self.proj_out = nn.Linear(d_model, out_dim)
        self.apply(_init_weights)

    def forward(self, x, mask=None):
        return self.proj_out(self.encoder(self.patch_embedding(x, mask)))


class EnhancedSpectralProjection(nn.Module):
    def __init__(self, patch_size, d_model, sampling_rate=200, dropout=0.1):
        super().__init__()
        self.patch_size    = patch_size
        self.sampling_rate = sampling_rate
        n_freq = patch_size // 2 + 1

        self.register_buffer('band_ranges', torch.tensor([
            [0.5,  4.0],
            [4.0,  8.0],
            [8.0,  13.0],
            [13.0, 30.0],
            [30.0, 50.0],
        ]))
        self.band_weights = nn.Parameter(torch.ones(5))

        dim1 = d_model // 3
        dim2 = d_model // 3
        dim3 = d_model - dim1 - dim2
        self.mag_proj   = nn.Linear(n_freq, dim1)
        self.phase_proj = nn.Linear(n_freq, dim2)
        self.power_proj = nn.Linear(n_freq, dim3)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        spectral  = torch.fft.rfft(x, dim=-1, norm='ortho')
        magnitude = torch.abs(spectral)
        phase     = torch.angle(spectral)
        power     = magnitude ** 2

        band_w    = F.softmax(self.band_weights, dim=0)
        freq_bins = torch.fft.rfftfreq(self.patch_size, 1 / self.sampling_rate).to(x.device)
        weights   = torch.ones_like(magnitude)
        for i, (low, high) in enumerate(self.band_ranges):
            weights[:, (freq_bins >= low) & (freq_bins < high)] = band_w[i]

        emb = torch.cat([self.mag_proj(magnitude * weights),
                         self.phase_proj(phase),
                         self.power_proj(power * weights)], dim=-1)
        return self.dropout(self.norm(emb))


class ChannelSelfAttentionPositionalEncoding(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, num_heads=heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, P, D = x.shape
        x_in = x.permute(0, 2, 1, 3).reshape(B * P, C, D)
        attn_out, _ = self.attn(x_in, x_in, x_in)
        out = self.norm(x_in + self.dropout(attn_out))
        return out.view(B, P, C, D).permute(0, 2, 1, 3)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, param):
        super().__init__()
        self.d_model    = d_model
        self.param      = param
        self.patch_size = 100

        self.mask_token           = nn.Parameter(torch.randn(in_dim) * 0.02)
        self.channel_pos_encoding = ChannelSelfAttentionPositionalEncoding(d_model)
        self.spectral_encoder     = EnhancedSpectralProjection(in_dim, d_model)

        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
        )

    def forward(self, x, mask=None):
        x = x[:, SELECTED_CHANNELS, :] if self.param.use_selected_channels else x[:, :-1, :]
        x = x * 1e4

        if not hasattr(self, 'hann_win') or self.hann_win.shape[-1] != x.shape[-1]:
            self.register_buffer('hann_win',
                                 torch.hann_window(x.shape[-1], device=x.device).view(1, 1, -1),
                                 persistent=False)
        x = F.layer_norm(x * self.hann_win, x.shape[-1:])

        B, C, T = x.shape
        x = x[:, :, :(T // self.patch_size) * self.patch_size].view(B, C, -1, self.patch_size)
        P = x.shape[2]

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1).expand_as(x) == 1,
                            self.mask_token.view(1, 1, 1, -1).expand_as(x),
                            x.clone())

        x_flat    = x.contiguous().view(B, 1, C * P, self.patch_size)
        patch_emb = self.proj_in(x_flat).permute(0, 2, 1, 3).contiguous().view(B, C, P, -1)

        spectral_emb = self.spectral_encoder(x_flat.view(B * C * P, self.patch_size)).view(B, C, P, -1)
        patch_emb    = patch_emb + spectral_emb

        return self.channel_pos_encoding(patch_emb)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model  = CBraMod(in_dim=100, out_dim=100, d_model=100,
                     dim_feedforward=400, seq_len=8, n_layer=12, nhead=4).to(device)
    a = torch.randn(8, 29, 8, 100).to(device)
    print(model(a).shape)
