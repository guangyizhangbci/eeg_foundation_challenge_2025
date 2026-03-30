import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_dynamic_mask_ratio(epoch, max_ratio=0.9, min_ratio=0.5,
                            warmup_epochs=5, decay_epochs=100, decay_factor=0.95):
    if epoch < warmup_epochs:
        return max_ratio * (epoch / warmup_epochs)
    return max(max_ratio * (decay_factor ** ((epoch - warmup_epochs) / decay_epochs)), min_ratio)


def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    return torch.bernoulli(
        torch.full((bz, ch_num, patch_num), mask_ratio, device=device)
    ).long()


class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss().to(self.device)
        self.writer = SummaryWriter(log_dir='/home/local/PARTNERS/gz005/tensorboard/pretrain')

        if params.parallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(8)))

        summary(self.model, input_size=(8, 29, 8, 100))
        macs, n_params = get_model_complexity_info(self.model, (29, 8, 100),
                                                   as_strings=True, print_per_layer_stat=True)
        print(f"{'Computational complexity:':<30}  {macs}")
        print(f"{'Number of parameters:':<30}  {n_params}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.lr, weight_decay=params.weight_decay)
        self.optimizer_scheduler = self._build_scheduler(params)

    def _build_scheduler(self, params):
        n = self.data_length = len(self.data_loader)
        s = params.lr_scheduler
        if s == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40*n, eta_min=1e-5)
        if s == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999999999)
        if s == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5*n, gamma=0.5)
        if s == 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*n, 20*n, 30*n], gamma=0.1)
        if s == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=1e-3,
                step_size_up=n*5, step_size_down=n*2,
                mode='exp_range', gamma=0.9, cycle_momentum=False)
        raise ValueError(f"Unknown lr_scheduler: {s}")

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device) / 100

                if self.params.need_mask:
                    bz, ch_num, patch_num, _ = x.shape
                    mask_ratio = get_dynamic_mask_ratio(epoch)
                    mask = generate_mask(bz, ch_num, patch_num, mask_ratio, self.device)
                    y = self.model(x, mask=mask)
                    loss = self.criterion(y[mask == 1], x[mask == 1])
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.item())

            mean_loss = np.mean(losses)
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}: loss={mean_loss:.6f}, lr={lr:.6f}')
            self.writer.add_scalar('Loss/train', mean_loss, epoch)
            self.writer.add_scalar('LearningRate', lr, epoch)

            if mean_loss < best_loss:
                path = f'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss:.6f}.pth'
                torch.save(self.model.state_dict(), path)
                print(f'Model saved: {path}')
                best_loss = mean_loss

        self.writer.close()
