import torch
import torch.nn as nn
import torch.nn.functional as F

from criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder

SELECTED_CHANNELS = [4, 5, 6, 12, 30, 34, 36, 40, 41, 51, 52, 53, 54, 60,
                     78, 79, 85, 86, 91, 92, 97, 102, 104, 105, 109, 110, 111, 116, 117]


class CBraMod(nn.Module):
    def __init__(self, in_dim=100, out_dim=100, d_model=100, dim_feedforward=400,
                 seq_len=8, n_layer=24, nhead=4, param=None):
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


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, param):
        super().__init__()
        self.d_model    = d_model
        self.param      = param
        self.patch_size = 100

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
        )

        self.spectral_proj = nn.Sequential(
            nn.Linear(self.patch_size + 2, d_model),
            nn.Dropout(0.1),
        )

        self.positional_encoding = nn.Conv2d(
            d_model, d_model, kernel_size=(29, 3),
            stride=(1, 1), padding=(14, 1), groups=d_model,
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
            x = x.clone()
            x[mask == 1] = self.mask_encoding

        x_flat = x.contiguous().view(B, 1, C * P, self.patch_size)

        patch_emb = self.proj_in(x_flat).permute(0, 2, 1, 3).contiguous().view(B, C, P, -1)

        spectral  = torch.fft.rfft(x_flat.view(B * C * P, self.patch_size), dim=-1, norm='forward')
        spec_feat = torch.cat([torch.abs(spectral), torch.angle(spectral)], dim=-1).view(B, C, P, -1)
        patch_emb = patch_emb + self.spectral_proj(spec_feat)

        pos_emb   = self.positional_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return patch_emb + pos_emb


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
