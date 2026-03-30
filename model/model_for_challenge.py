import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.cbramod_v1     import CBraMod as CBraMod_v1
from models.cbramod_v2     import CBraMod as CBraMod_v2
from models.cbramod_v3     import CBraMod as CBraMod_v3
from models.cbramod_v4     import CBraMod as CBraMod_v4
from models.cbramod_v5     import CBraMod as CBraMod_v5

FOUNDATION_DIRS = {
    'v1': '/home/patrick/llm_project/foundation_ckpt_v1/epoch100_loss4.822174446417193e-07.pth',
    'v2': '/home/patrick/llm_project/foundation_ckpt_v2/epoch100_loss5.948982675363368e-07.pth',
    'v3': '/home/patrick/llm_project/foundation_ckpt_v3/epoch16_loss4.261688445694745e-06.pth',
    'v4': '/home/patrick/llm_project/foundation_ckpt_v4/epoch69_loss8.693559152561647e-07.pth',
    'v5': '/home/patrick/llm_project/foundation_ckpt_v5/epoch33_loss8.50480830649758e-07.pth',
}


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.backbone, self.foundation_dir, self.output_dim = self._build_backbone(param)

        if param.use_pretrained_weights:
            device = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(self.foundation_dir, map_location=device))

        self.backbone.proj_out = nn.Identity()
        n_channels = 29 if param.use_selected_channels else 128
        self.classifier = self._build_classifier(param, n_channels)

    def _build_backbone(self, param):
        if param.use_cbramod_v1:
            return (CBraMod_v1(in_dim=100, out_dim=100, d_model=100, dim_feedforward=400,
                               seq_len=8, n_layer=12, nhead=4, param=param),
                    FOUNDATION_DIRS['v1'], 100)
        if param.use_cbramod_v2:
            return (CBraMod_v2(in_dim=100, out_dim=100, d_model=100, dim_feedforward=400,
                               seq_len=8, n_layer=24, nhead=4, param=param),
                    FOUNDATION_DIRS['v2'], 100)
        if param.use_cbramod_v3:
            return (CBraMod_v3(in_dim=100, out_dim=100, d_model=100, dim_feedforward=400,
                               seq_len=8, n_layer=24, nhead=4, param=param),
                    FOUNDATION_DIRS['v3'], 100)
        if param.use_cbramod_v4:
            return (CBraMod_v4(in_dim=100, out_dim=100, d_model=100, dim_feedforward=800,
                               seq_len=8, n_layer=24, nhead=4, param=param),
                    FOUNDATION_DIRS['v4'], 100)
        if param.use_cbramod_v5:
            return (CBraMod_v5(in_dim=100, out_dim=100, d_model=100, dim_feedforward=800,
                               seq_len=8, n_layer=24, nhead=10, param=param),
                    FOUNDATION_DIRS['v5'], 100)
        return (CBraMod(in_dim=100, out_dim=100, d_model=100, dim_feedforward=400,
                        seq_len=8, n_layer=12, nhead=4, param=param),
                param.foundation_dir, 100)

    def _build_classifier(self, param, n_channels):
        D = self.output_dim
        if param.classifier == 'avgpooling_patch_reps':
            return nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(100, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        if param.classifier == 'all_patch_reps_onelayer':
            return nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(29 * 2 * 100, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        if param.classifier == 'all_patch_reps_twolayer':
            return nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(29 * 2 * 100, 100),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(100, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        # default: all_patch_reps
        return nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(n_channels * 2 * D, 256),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            Rearrange('b 1 -> (b 1)'),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
