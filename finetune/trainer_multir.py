import gc
import os
import random
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from evaluator import Evaluator
from datasets import challenge_1_dataset_cache_multir_ as challenge_1_dataset
from datasets import challenge_2_dataset_cache_multir_ as challenge_2_dataset

TIMESTAMP = datetime.now().strftime("%b-%d-%Y-time-%H-%M")

ALL_R_PATHS = [
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R1_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R2_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R3_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R4_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R5_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R6_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R7_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R8_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R9_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R10_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/NC_100_bdf_bids",
]

TEST_PATH  = "/home/patrick/llm_project/data/HBN/finetune/BIDS/R11_100_bdf_bids"
DATA_LENGTHS = {'Challenge-1': 116890, 'Challenge-2': 866769}

SELECTED_CHANNELS = [4, 5, 6, 12, 30, 34, 36, 40, 41, 51, 52, 53, 54, 60,
                     78, 79, 85, 86, 91, 92, 97, 102, 104, 105, 109, 110, 111, 116, 117]


class Trainer:
    def __init__(self, params, model):
        self.params = params
        self.model  = model.cuda()
        self.criterion = MSELoss().cuda()
        self.best_model_states = None

        backbone_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = not params.frozen
                backbone_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = self._build_optimizer(params, backbone_params, other_params)
        self.data_length = DATA_LENGTHS[params.downstream_dataset]
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=params.epochs * self.data_length, eta_min=1e-6)
        print(self.model)

    def _build_optimizer(self, params, backbone_params, other_params):
        param_groups = (
            [{'params': backbone_params, 'lr': params.lr},
             {'params': other_params,   'lr': params.lr * 5}]
            if params.multi_lr else self.model.parameters()
        )
        if params.optimizer == 'AdamW':
            return torch.optim.AdamW(param_groups, lr=params.lr,
                                     weight_decay=params.weight_decay)
        return torch.optim.SGD(param_groups, lr=params.lr,
                               momentum=0.9, weight_decay=params.weight_decay)

    def _get_combined_loader(self, paths):
        if self.params.downstream_dataset == 'Challenge-1':
            return challenge_1_dataset.get_combined_dataloader(
                paths, batch_size=256, num_workers=8, use_cache=True)
        return challenge_2_dataset.get_combined_dataloader(
            paths, batch_size=256, num_workers=8, use_cache=True)

    def _get_test_loader(self):
        if self.params.downstream_dataset == 'Challenge-1':
            return challenge_1_dataset.get_data_loader(TEST_PATH, use_cache=True)
        return challenge_2_dataset.get_data_loader(TEST_PATH, use_cache=True)

    def _close_datasets(self, datasets):
        if self.params.downstream_dataset == 'Challenge-1':
            challenge_1_dataset.close_datasets(datasets)
        else:
            challenge_2_dataset.close_datasets(datasets)

    def _free(self, *objs):
        for obj in objs: del obj
        torch.cuda.empty_cache()
        gc.collect()

    def _get_version(self):
        p = self.params
        if p.use_cbramod_v1:   return 'v1'
        if p.use_cbramod_v2:   return 'v2'
        if p.use_cbramod_v3:   return 'v3'
        if p.use_cbramod_v4:   return 'v4'
        if p.use_cbramod_v6:   return 'v6'
        return 'default'

    def filter_bad_eeg(self, x):
        x_sel = x[:, SELECTED_CHANNELS, :] * 1e4
        bad_channels_per_sample = (x_sel.abs() > 3).any(dim=-1).sum(dim=-1)
        return bad_channels_per_sample <= 3

    def mask_bad_channels(self, x):
        x = x * 1e4
        keep_mask = (~(x.abs() > 5).any(dim=-1)).float().unsqueeze(-1)
        return x * keep_mask

    def train_for_regression(self):
        epoch_release_history = []

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()

            shuffled = ALL_R_PATHS.copy()
            random.shuffle(shuffled)
            release_names = [os.path.basename(p).split('_')[0] for p in shuffled]
            print(f"\nEpoch {epoch+1}/{self.params.epochs} — release order: {', '.join(release_names)}")
            epoch_release_history.append(release_names)

            loader, datasets = self._get_combined_loader(shuffled)
            losses = []

            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()
                x, y = x.cuda().float(), y.cuda().float()
                loss = self.criterion(self.model(x).view(-1, 1), y.view(-1, 1))
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.item())

            print(f"Epoch {epoch+1} loss: {np.mean(losses):.5f}  "
                  f"({(timer() - start_time) / 60:.2f} min)")

            self._close_datasets(datasets)
            self._free(loader, datasets)

            self._run_test_and_save(epoch, epoch_release_history)

    def _run_test_and_save(self, epoch, epoch_release_history):
        with torch.no_grad():
            loader    = self._get_test_loader()
            evaluator = Evaluator(self.params, loader)
            corrcoef, r2, rmse = evaluator.get_metrics_for_regression(self.model)
            print(f"Test — corrcoef: {corrcoef:.5f}, r2: {r2:.5f}, rmse: {rmse:.5f}")

            version  = self._get_version()
            channels = 'selected_channels' if self.params.use_selected_channels else 'all_channels'
            save_dir = os.path.join(self.params.model_dir, self.params.downstream_dataset,
                                    version, channels, TIMESTAMP)
            os.makedirs(save_dir, exist_ok=True)

            model_path = os.path.join(
                save_dir,
                f"epoch{epoch+1}_corrcoef_{corrcoef:.5f}_r2_{r2:.5f}_rmse_{rmse:.5f}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

            args_path = os.path.join(save_dir, "args.txt")
            with open(args_path, 'w') as f:
                f.write(f"{'='*70}\nTRAINING ARGUMENTS\n{'='*70}\n\n"
                        f"Timestamp: {TIMESTAMP}\n"
                        f"Releases per epoch: {len(ALL_R_PATHS)}\n"
                        f"Training mode: all releases combined and shuffled\n\n")
                for k, v in vars(self.params).items():
                    f.write(f"{k}: {v}\n")
                f.write(f"\n{'='*70}\nEPOCH RELEASE ORDER\n{'='*70}\n\n")
                for ep, releases in enumerate(epoch_release_history, 1):
                    f.write(f"Epoch {ep}: {', '.join(releases)}\n")

            print(f"Args saved: {args_path}")
            self._free(loader, evaluator)
