import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam

cudnn.benchmark = True
cudnn.deterministic = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2" # version check

import vision_transformer
from mia_lib.mia_util import get_tinyimagenet_dataloaders, MultiCropWrapper, get_cifar100_dataloaders, get_cifar10_dataloaders
from mia_lib.mia_model import Adversary
from mia_lib.mia_train import mia_train
from mia_lib.mia_eval import mia_evaluate



def save_checkpoints(args, attacker, epoch, save_name):
    obj = {
        'net':attacker.state_dict(),
        'epoch': epoch
    }
    torch.save(obj, os.path.join(args.output_dir, save_name))

def load_checkpoints(args, attacker, save_name):
    obj = torch.load(os.path.join(args.output_dir, save_name))
    attacker.load_state_dict(obj['net'])
    return attacker
    


def get_args_parser():

    parser = argparse.ArgumentParser('Self-supervised learning & Membership inference attack')
    parser.add_argument('--data', type=str, default='tinyimagenet', help='dataset used in pretrain and attack')
    parser.add_argument('--datadir',type=str, default='../data/tiny-imagenet-200/', help='especially for tinyimagenet')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='vit_base_patch16')
    parser.add_argument('--img_size',type=int, default=64)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100, help='attacker training epoch')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--global_crops_scale', type=str, default='0.2,1.0')
    parser.add_argument('--local_crops_scale', type=str, default='0.08,0.2')
    parser.add_argument('--local_crops_number', type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--feature', type=str, default='both', help='uniform, gassian and both for training attacker')
    parser.add_argument('--output_dir',type=str, default='./output_dir', help='place to save checkpoint')
    return parser


def main(args):
    
    print('prepare global crops scale, and local crops scale')
    args.global_crops_scale = tuple(eval(i) for i in args.global_crops_scale.split(','))
    args.local_crops_scale = tuple(eval(i) for i in args.local_crops_scale.split(','))

    print('mdkir output dir')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print("obtain dataset")
    if args.data == 'tinyimagenet':
        train_loader, test_loader, test_loader, trainknownset_loader, train_infset_loader, testknownset_loader, test_infset_loader = get_tinyimagenet_dataloaders(args)
    elif args.data == 'cifar100':
        train_loader, test_loader, test_loader, trainknownset_loader, train_infset_loader, testknownset_loader, test_infset_loader = get_cifar100_dataloaders(args)
    elif args.data == 'cifar10':
        train_loader, test_loader, test_loader, trainknownset_loader, train_infset_loader, testknownset_loader, test_infset_loader = get_cifar10_dataloaders(args)
    else:
        NotImplementedError()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    print('instantiate adversary')

    attacker = Adversary(args.local_crops_number*2 if args.feature=='both' else args.local_crops_number).to(device)
    optimizer = Adam(attacker.parameters(), lr = args.lr, weight_decay=args.l2)

    print('instantiate victim model')
    model = vision_transformer.__dict__[args.model](img_size=[args.img_size])

    ## remove backbone
    param = torch.load(args.model_path,map_location='cpu')['student']
    for k in list(param.keys()):
        if 'backbone' in k:
            param['.'.join(k.split('.')[1:])] = param[k]
            param.pop(k)

    msg = model.load_state_dict(param, strict=False)
    model = MultiCropWrapper(model) 
    
    print(msg)
    model = model.to(device) 
    model.eval()

    size = len(testknownset_loader)
    
    private_testset = enumerate(zip(train_infset_loader, test_infset_loader))
    mia_evaluate(args, model, attacker, device, private_testset, is_test_set=False)
    best_acc = 0

    for epoch in range(args.epochs):

        private_trainset = enumerate(zip(trainknownset_loader, testknownset_loader))
        mia_train(args, model, attacker, device, private_trainset, optimizer, size)

        private_testset = enumerate(zip(train_infset_loader, test_infset_loader))
        acc = mia_evaluate(args, model, attacker, device, private_testset, is_test_set=False)
        
        if acc >= best_acc:
            best_acc = acc
            save_checkpoints(args, attacker, epoch, 'best.pth')

        save_checkpoints(args, attacker, epoch, 'last.pth')

    # test
    private_testset = enumerate(zip(train_infset_loader, test_infset_loader))
    attacker = load_checkpoints(args, attacker, 'best.pth')
    acc = mia_evaluate(args, model, attacker, device, private_testset, is_test_set=True)

    print('finish')



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
