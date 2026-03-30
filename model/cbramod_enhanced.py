import torch
import torch.nn as nn
import torch.nn.functional as F

from criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder


class CBraMod(nn.Module):
    def __init__(self, in_dim=100, out_dim=100, d_model=100, dim_feedforward=800,
                 seq_len=8, n_layer=24, nhead=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
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
    def __init__(self, patch_size, d_model, sampling_rate=100, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.sampling_rate = sampling_rate
        n_freq = patch_size // 2 + 1

        self.register_buffer('band_ranges', torch.tensor([
            [0.5,  4.0],   # Delta
            [4.0,  8.0],   # Theta
            [8.0,  13.0],  # Alpha
            [13.0, 30.0],  # Beta
            [30.0, 50.0],  # Gamma
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

        band_w   = F.softmax(self.band_weights, dim=0)
        freq_bins = torch.fft.rfftfreq(self.patch_size, 1 / self.sampling_rate).to(x.device)
        weights  = torch.ones_like(magnitude)
        for i, (low, high) in enumerate(self.band_ranges):
            weights[:, (freq_bins >= low) & (freq_bins < high)] = band_w[i]

        magnitude = magnitude * weights
        power     = power     * weights

        emb = torch.cat([self.mag_proj(magnitude),
                         self.phase_proj(phase),
                         self.power_proj(power)], dim=-1)
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
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(in_dim) * 0.02)
        self.channel_pos_encoding = ChannelSelfAttentionPositionalEncoding(d_model)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
        )
        self.spectral_encoder = EnhancedSpectralProjection(in_dim, d_model)
        self.spectral_gate    = nn.Parameter(torch.ones(1) * 0.5)

    def robust_standardize(self, x, eps=1e-6):
        median = x.median(dim=-1, keepdim=True)[0]
        mad    = torch.median(torch.abs(x - median), dim=-1, keepdim=True)[0]
        return (x - median) / (1.4826 * mad + eps)

    def forward(self, x, mask=None):
        x = self.robust_standardize(x)
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1).expand_as(x) == 1,
                            self.mask_token.view(1, 1, 1, -1).expand_as(x),
                            x.clone())

        x_reshaped = x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)

        patch_emb = self.proj_in(x_reshaped)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, -1)

        spectral_emb = self.spectral_encoder(x_reshaped.view(bz * ch_num * patch_num, patch_size))
        spectral_emb = spectral_emb.view(bz, ch_num, patch_num, -1)

        patch_emb = patch_emb + torch.sigmoid(self.spectral_gate) * spectral_emb
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = CBraMod(in_dim=100, out_dim=100, d_model=100,
                     dim_feedforward=400, seq_len=8, n_layer=12, nhead=4).to(device)
    a = torch.randn(8, 29, 8, 100).to(device)
    print(model(a).shape)
