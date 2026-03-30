import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGNeX

SELECTED_CHANNELS = [4, 5, 6, 12, 30, 34, 36, 40, 41, 51, 52, 53, 54, 60,
                     78, 79, 85, 86, 91, 92, 97, 102, 104, 105, 109, 110, 111, 116, 117]


class CBraMod(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.model = EEGNeX(n_chans=29, n_outputs=1, n_times=200, sfreq=100)

    def forward(self, x, mask=None):
        x = x[:, SELECTED_CHANNELS, :] if self.param.use_selected_channels else x[:, :-1, :]
        x = F.layer_norm(x * 1e4, x.shape[-1:])

        if not hasattr(self, 'hann_win') or self.hann_win.shape[-1] != x.shape[-1]:
            self.register_buffer('hann_win',
                                 torch.hann_window(x.shape[-1], device=x.device).view(1, 1, -1),
                                 persistent=False)
        return self.model(x * self.hann_win)
