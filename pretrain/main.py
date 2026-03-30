import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod_enhanced import CBraMod

from trainer import Trainer
import json
import os
from datetime import datetime


lmdb_dirs = [
    '/home/patrick/llm_project/data/HBN/pretraining/R1.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R2.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R3.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R4.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R5.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R6.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R7.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R8.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R9.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R10.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/R11.lmdb/',
    '/home/patrick/llm_project/data/HBN/pretraining/NC.lmdb/',
    # Add more LMDB folders here
]



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=100, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=100, help='out_dim')
    parser.add_argument('--d_model', type=int, default=100, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=8, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=4, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--model_dir',   type=str,   default='/home/patrick/llm_project/foundation_ckpt_v4', help='model_dir')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    # pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    # print(len(pretrained_dataset))



    def save_args(args, save_dir):
        """
        Save arguments to a JSON file
        
        Args:
            args: argparse.Namespace object
            save_dir: directory to save the args file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Save to JSON file
        args_file = os.path.join(save_dir, 'args.json')
        with open(args_file, 'w') as f:
            json.dump(args_dict, f, indent=4)
        
        print(f"✓ Arguments saved to {args_file}")
        
        # Also save a human-readable text version
        txt_file = os.path.join(save_dir, 'args.txt')
        with open(txt_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Training Arguments - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            for key, value in args_dict.items():
                f.write(f"{key:20s}: {value}\n")
        
        print(f"✓ Arguments saved to {txt_file}")

    save_args(params, params.model_dir)

    all_datasets = [PretrainingDataset(d) for d in lmdb_dirs]
    pretrained_dataset = ConcatDataset(all_datasets)

    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )
    model = CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
        params.nhead
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
