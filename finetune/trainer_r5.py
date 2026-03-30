import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


class Trainer_r5:
    def __init__(self, params, data_loader, model):
        self.params      = params
        self.data_loader = data_loader
        self.model       = model.cuda()
        self.criterion   = MSELoss().cuda()
        self.val_eval    = Evaluator(params, data_loader['val'])
        self.test_eval   = Evaluator(params, data_loader['test'])

        if params.use_regression_norm:
            params.y_mean, params.y_std = data_loader['norm']

        backbone_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = not params.frozen
                backbone_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = self._build_optimizer(params, backbone_params, other_params)
        self.data_length = len(data_loader['train'])
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

    def _get_version(self):
        p = self.params
        if p.use_200hz:       return '200hz'
        if p.use_small_model: return 'small'
        if p.use_cbramod_v1:  return 'v1'
        if p.use_cbramod_v2:  return 'v2'
        if p.use_cbramod_v3:  return 'v3'
        return 'default'

    def train_for_regression(self):
        best_rmse, best_epoch = 1.0, 0
        best_model_states     = None

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x, y = x.cuda().float(), y.cuda().float()
                pred = self.model(x).view(-1, 1)
                y    = y.view(-1, 1)
                loss = self.criterion(
                    pred, (y - self.params.y_mean) / self.params.y_std
                    if self.params.use_regression_norm else y
                )
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.item())

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}: loss={np.mean(losses):.5f}, "
                      f"val corrcoef={corrcoef:.5f}, r2={r2:.5f}, rmse={rmse:.5f}, "
                      f"lr={lr:.5f}, time={(timer()-start_time)/60:.2f}min")

                if rmse < best_rmse:
                    best_rmse, best_epoch = rmse, epoch + 1
                    best_model_states = copy.deepcopy(self.model.state_dict())
                    print(f"  Best model updated — corrcoef={corrcoef:.5f}, r2={r2:.5f}, rmse={rmse:.5f}")

        self.model.load_state_dict(best_model_states)
        with torch.no_grad():
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print(f"Test — corrcoef: {corrcoef:.5f}, r2: {r2:.5f}, rmse: {rmse:.5f}")

            version  = self._get_version()
            save_dir = os.path.join(self.params.model_dir, self.params.downstream_dataset, version)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(
                save_dir,
                f"epoch{best_epoch}_corrcoef_{corrcoef:.5f}_r2_{r2:.5f}_rmse_{rmse:.5f}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
