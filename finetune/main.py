import argparse
import random

import numpy as np
import torch

from datasets import challenge_1_dataset, challenge1_r5_dataset, challenge2_r5_dataset
# from finetune_trainer import Trainer
from trainer_multir import Trainer
from trainer_r5 import Trainer_r5

import sys

from models import model_for_challenge_1


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--use-regression-norm', action='store_true', help='Apply normalization to regression labels')

    parser.add_argument('--downstream_dataset', type=str, default='Challenge-1',
                        help='[Challenge-2, Challenge-1]')
    
    parser.add_argument('--use-r-5-only', action='store_true', help='Use only R5 data')
    parser.add_argument('--use-selected-channels', action='store_true', help='Use only selected channels')

    # parser.add_argument('--datasets_dir', type=str,
    #                     default='/home/patrick/Documents/EEG_LLM/data/BigDownstream/Faced/processed',
    #                     help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/home/patrick/llm_project/ft_ckpt', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=12, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use-pretrained-weights', type=lambda x: x.lower() == 'true', 
                        default=True, help='use_pretrained_weights')
 
    parser.add_argument('--use-200hz', action='store_true', help='use_200hz')
    
    parser.add_argument('--foundation_dir', type=str,
                        default='/home/patrick/llm_project/foundation_ckpt/epoch100_loss3.215673132217489e-07.pth',
                        help='foundation_dir')
    parser.add_argument('--foundation_dir_200hz', type=str,
                        default='/home/patrick/llm_project/foundation_ckpt/pretrained_weights_cbramod_200hz.pth',
                        help='foundation_dir')

    parser.add_argument('--use-cbramod-v1',  action='store_true', help='Use New architecture')
    parser.add_argument('--use-cbramod-v2',  action='store_true', help='Use New architecture')
    parser.add_argument('--use-cbramod-v3',  action='store_true', help='Use New architecture')
    parser.add_argument('--use-cbramod-v4',  action='store_true', help='Use New architecture')
    parser.add_argument('--use-cbramod-v5',  action='store_true', help='Use New architecture')
    parser.add_argument('--use-cbramod-v6',  action='store_true', help='Use New architecture')




    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    if params.downstream_dataset == 'Challenge-1':

        model = model_for_challenge_1.Model(params)

        if params.use_r_5_only:
            data_loader = challenge1_r5_dataset.get_data_loader()
            t = Trainer_r5(params, data_loader, model)

        else: 
            t = Trainer(params, model)

        t.train_for_regression()

    elif params.downstream_dataset == 'Challenge-2':

        model = model_for_challenge_1.Model(params)
        if params.use_r_5_only:
            data_loader = challenge2_r5_dataset.get_data_loader()
            t = Trainer_r5(params, data_loader, model)

        else: 
            t = Trainer(params, model)

        t.train_for_regression()

    else: 
        print('Select Task')


    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
