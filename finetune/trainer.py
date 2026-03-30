import gc
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator
from datasets import challenge_1_dataset_cache as challenge_1_dataset
from datasets import challenge_2_dataset_cache as challenge_2_dataset

TIMESTAMP = datetime.now().strftime("%b-%d-%Y-time-%H-%M")

R_PATHS = [
    "import gc
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator
from datasets import challenge_1_dataset_cache as challenge_1_dataset
from datasets import challenge_2_dataset_cache as challenge_2_dataset

TIMESTAMP = datetime.now().strftime("%b-%d-%Y-time-%H-%M")

R_PATHS = [
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R1_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R2_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R3_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R4_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R6_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R7_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R8_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R9_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R10_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/NC_100_bdf_bids",
]

TEST_PATH = "/home/patrick/llm_project/data/HBN/finetune/BIDS/R11_100_bdf_bids"

DATA_LENGTHS = {'Challenge-1': 116890, 'Challenge-2': 866769}


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
        if params.multi_lr:
            param_groups = [{'params': backbone_params, 'lr': params.lr},
                            {'params': other_params,   'lr': params.lr * 5}]
        else:
            param_groups = self.model.parameters()

        if params.optimizer == 'AdamW':
            return torch.optim.AdamW(param_groups, lr=params.lr,
                                     weight_decay=params.weight_decay)
        return torch.optim.SGD(param_groups, lr=params.lr,
                               momentum=0.9, weight_decay=params.weight_decay)

    def _get_loader(self, path, **kwargs):
        if self.params.downstream_dataset == 'Challenge-1':
            return challenge_1_dataset.get_data_loader(path, **kwargs)
        return challenge_2_dataset.get_data_loader(path)

    def _free(self, *objs):
        for obj in objs:
            del obj
        torch.cuda.empty_cache()
        gc.collect()

    def _get_version(self):
        p = self.params
        if p.use_200hz:         return '200hz'
        if p.use_small_model:   return 'small'
        if p.use_cbramod_v1:    return 'v1'
        if p.use_cbramod_v2:    return 'v2'
        if p.use_cbramod_v3:    return 'v3'
        if p.use_cbramod_v4:    return 'v4'
        return 'default'

    def train_for_regression(self):
        for epoch in range(self.params.epochs):
            self.model.train()
            losses, r_losses = [], {i: [] for i in range(len(R_PATHS))}

            for r_idx, r_path in enumerate(R_PATHS):
                print(f"Loading R{r_idx+1} from {r_path}")
                loader = self._get_loader(r_path, batch_size=512, use_cache=True)

                for x, y in tqdm(loader, mininterval=10, desc=f"R{r_idx+1}"):
                    self.optimizer.zero_grad()
                    x, y = x.cuda().float(), y.cuda().float()
                    pred = self.model(x).view(-1, 1)
                    loss = self.criterion(pred, y.view(-1, 1))
                    loss.backward()
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    val = loss.item()
                    losses.append(val)
                    r_losses[r_idx].append(val)

                print(f"R{r_idx+1} avg loss: {np.mean(r_losses[r_idx]):.5f}")
                self.optimizer_scheduler.step()
                self._free(loader)

            print(f"Epoch {epoch+1} avg loss: {np.mean(losses):.5f}")
            self._run_test_and_save(epoch)

    def _run_test_and_save(self, epoch):
        with torch.no_grad():
            loader    = self._get_loader(TEST_PATH, use_cache=True)
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
                f.write(f"{'='*70}\nTRAINING ARGUMENTS\n{'='*70}\n\nTimestamp: {TIMESTAMP}\n\n")
                for k, v in vars(self.params).items():
                    f.write(f"{k}: {v}\n")
            print(f"Args saved: {args_path}")

            self._free(loader, evaluator)/llm_project/data/HBN/finetune/BIDS/R1_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R2_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R3_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R4_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R6_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R7_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R8_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R9_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/R10_100_bdf_bids",
    "/home/patrick/llm_project/data/HBN/finetune/BIDS/NC_100_bdf_bids",
]

TEST_PATH = "/home/patrick/llm_project/data/HBN/finetune/BIDS/R11_100_bdf_bids"

DATA_LENGTHS = {'Challenge-1': 116890, 'Challenge-2': 866769}


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
        if params.multi_lr:
            param_groups = [{'params': backbone_params, 'lr': params.lr},
                            {'params': other_params,   'lr': params.lr * 5}]
        else:
            param_groups = self.model.parameters()

        if params.optimizer == 'AdamW':
            return torch.optim.AdamW(param_groups, lr=params.lr,
                                     weight_decay=params.weight_decay)
        return torch.optim.SGD(param_groups, lr=params.lr,
                               momentum=0.9, weight_decay=params.weight_decay)

    def _get_loader(self, path, **kwargs):
        if self.params.downstream_dataset == 'Challenge-1':
            return challenge_1_dataset.get_data_loader(path, **kwargs)
        return challenge_2_dataset.get_data_loader(path)

    def _free(self, *objs):
        for obj in objs:
            del obj
        torch.cuda.empty_cache()
        gc.collect()

    def _get_version(self):
        p = self.params
        if p.use_200hz:         return '200hz'
        if p.use_small_model:   return 'small'
        if p.use_cbramod_v1:    return 'v1'
        if p.use_cbramod_v2:    return 'v2'
        if p.use_cbramod_v3:    return 'v3'
        if p.use_cbramod_v4:    return 'v4'
        return 'default'

    def train_for_regression(self):
        for epoch in range(self.params.epochs):
            self.model.train()
            losses, r_losses = [], {i: [] for i in range(len(R_PATHS))}

            for r_idx, r_path in enumerate(R_PATHS):
                print(f"Loading R{r_idx+1} from {r_path}")
                loader = self._get_loader(r_path, batch_size=512, use_cache=True)

                for x, y in tqdm(loader, mininterval=10, desc=f"R{r_idx+1}"):
                    self.optimizer.zero_grad()
                    x, y = x.cuda().float(), y.cuda().float()
                    pred = self.model(x).view(-1, 1)
                    loss = self.criterion(pred, y.view(-1, 1))
                    loss.backward()
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    val = loss.item()
                    losses.append(val)
                    r_losses[r_idx].append(val)

                print(f"R{r_idx+1} avg loss: {np.mean(r_losses[r_idx]):.5f}")
                self.optimizer_scheduler.step()
                self._free(loader)

            print(f"Epoch {epoch+1} avg loss: {np.mean(losses):.5f}")
            self._run_test_and_save(epoch)

    def _run_test_and_save(self, epoch):
        with torch.no_grad():
            loader    = self._get_loader(TEST_PATH, use_cache=True)
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
                f.write(f"{'='*70}\nTRAINING ARGUMENTS\n{'='*70}\n\nTimestamp: {TIMESTAMP}\n\n")
                for k, v in vars(self.params).items():
                    f.write(f"{k}: {v}\n")
            print(f"Args saved: {args_path}")

            self._free(loader, evaluator)
