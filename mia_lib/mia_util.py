
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import math
import torch.nn.functional as F

def generate_gassian_ditribution(N, length):
    noise = torch.normal(0.0, 1.0, size=(N,length)) 
    noise = noise.sort(dim=1)[0]
    # print(noise.shape)
    # print(noise)
    noise = (-(noise-noise.mean(dim=1, keepdim=True))**2/(2*noise.std(dim=1,  keepdim=True)**2)).exp()/(math.sqrt(2*math.pi)*noise.std(dim=1,  keepdim=True))
    noise = noise/noise.sum(dim=1, keepdim=True)
    # print(noise)
    return noise

def obtain_membership_feature(global_feature_map, local_feature_vectors, feature_type='both'):
    ## feature calculation to obtain the score distribution 
    B, L, D = global_feature_map.shape

    assert local_feature_vectors.shape[0]%global_feature_map.shape[0]==0
    local_feature_vectors = local_feature_vectors.view(B, -1, D)

    sim_score = torch.einsum('nik,njk->nij',[local_feature_vectors, global_feature_map])  # B N L
    assert sim_score.shape[1:]==(local_feature_vectors.shape[1], global_feature_map.shape[1])
    logit_score = F.log_softmax(sim_score,dim=2)
    
    _, N, L = sim_score.shape

    if feature_type=='both':
        ## calculate the engery with uniform distribution
        uniform = torch.ones_like(sim_score).to(sim_score.device)/L
        uniform_score = (uniform * ((uniform+1e-6).log()-logit_score)).sum(dim=2) # rank 
        sorted_uniform_score = torch.sort(uniform_score, dim=1, descending=True)[0] # B N

        ## calculate the engery with gassian distribution
        gassian = generate_gassian_ditribution(N,L).to(sim_score.device).unsqueeze(dim=0).repeat(B,1,1)
        gassian_score = (gassian * ((gassian+1e-6).log()-logit_score)).sum(dim=2) 
        sorted_gassian_score = torch.sort(gassian_score, dim=1, descending=True)[0]  # B N

        feature = torch.cat([sorted_uniform_score, sorted_gassian_score], dim=1)  # B 2N

        return feature

    elif feature_type=='uniform':
        ## calculate the engery with uniform distribution
        uniform = torch.ones_like(sim_score).to(sim_score.device)/L
        uniform_score = (uniform * ((uniform+1e-6).log()-logit_score)).sum(dim=2) # rank 
        sorted_uniform_score = torch.sort(uniform_score, dim=1, descending=True)[0] # B N

        return sorted_uniform_score
    
    elif feature_type=='gassian':
        gassian = generate_gassian_ditribution(N,L).to(sim_score.device).unsqueeze(dim=0).repeat(B,1,1)
        gassian_score = (gassian * ((gassian+1e-6).log()-logit_score)).sum(dim=2) 
        sorted_gassian_score = torch.sort(gassian_score, dim=1, descending=True)[0]  # B N

        return sorted_gassian_score

    else:
        NotImplementedError()

## this code is borrowed from dino
class DataAugmentation(object):
    def __init__(self, img_size, part_size, img_mean, img_std, global_crops_scale, local_crops_scale, local_crops_number, is_train=True):

        self.normlize = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std)])
        
        self.is_train  = is_train
        # global augmentation --> whole image
        self.global_transfo_t = transforms.Resize(size=img_size, interpolation=3)
        self.global_transfo = transforms.Compose([transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=3), 
                transforms.RandomHorizontalFlip(p=0.5)])  # 3 is bicubic

        # local small crops --> part
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([transforms.RandomResizedCrop(part_size, scale=local_crops_scale, interpolation=3),
                            transforms.RandomHorizontalFlip(p=0.5)])

    def __call__(self, image):
        crops = []
        if self.is_train:
            crop = self.global_transfo(image)
            crops.append(self.normlize(crop))
        else:
            crop = self.global_transfo_t(image)
            crops.append(self.normlize(crop))

        for _ in range(self.local_crops_number):
            crops.append(self.normlize(self.local_transfo(crop)))

        return crops

## this code is borrowed from dino
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0 
        output = []
        for end_idx in idx_crops:

            _out = self.backbone.forward_features(torch.cat(x[start_idx: end_idx]), start_idx==0)

            output.append(_out)  # B*N 1 C
            start_idx = end_idx

        return output  ## two elements [one  global feature_map and some local vectors]

def get_tinyimagenet_dataloaders(args):
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
    

    tinyimage_mean=[0.485, 0.456, 0.406]
    tinyimage_std=[0.229, 0.224, 0.225]

    train_transform = DataAugmentation(args.img_size, args.img_size//4, tinyimage_mean, tinyimage_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=True)

    test_transform = DataAugmentation(args.img_size, args.img_size//4, tinyimage_mean, tinyimage_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=False)

    trainset = datasets.ImageFolder(traindir, train_transform)
    testset = datasets.ImageFolder(valdir, test_transform)

    trainset_length = len(trainset)
    random.seed(42)
    trainset_index = random.sample(range(trainset_length), int(trainset_length / 2))
    # unseen part to test inference model
    train_inf_test_index = list(set(range(trainset_length)).difference(set(trainset_index)))
    knowntrainset = torch.utils.data.Subset(trainset, trainset_index)
    infset = torch.utils.data.Subset(trainset, train_inf_test_index)

    infset_loader = torch.utils.data.DataLoader(infset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    knownset_loader = torch.utils.data.DataLoader(knowntrainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=8)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    random.seed(18)
    testset_length = len(testset)
    testset_index = random.sample(range(testset_length), int(testset_length / 2))
    referenceset = torch.utils.data.Subset(testset, testset_index)
    test_inf_test_index = list(set(range(testset_length)).difference(set(testset_index)))
    test_infset = torch.utils.data.Subset(testset, test_inf_test_index)

    reference_loader = torch.utils.data.DataLoader(referenceset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8)
    test_infset_loader = torch.utils.data.DataLoader(test_infset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, test_loader, knownset_loader, infset_loader, reference_loader, test_infset_loader


def get_cifar100_dataloaders(args):

    cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # cifar100_mean=[0.485, 0.456, 0.406]
    # cifar100_std=[0.229, 0.224, 0.225]

    train_transform = DataAugmentation(args.img_size, args.img_size//2, cifar100_mean, cifar100_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=True)

    test_transform = DataAugmentation(args.img_size, args.img_size//2, cifar100_mean, cifar100_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=False)

    trainset = datasets.CIFAR100(root='../data/cifar100', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR100(root='../data/cifar100', train=False, download=True, transform=test_transform)

    trainset_length = len(trainset)
    random.seed(42)
    trainset_index = random.sample(range(trainset_length), int(trainset_length / 2))
    # unseen part to test inference model
    train_inf_test_index = list(set(range(trainset_length)).difference(set(trainset_index)))
    knowntrainset = torch.utils.data.Subset(trainset, trainset_index)
    infset = torch.utils.data.Subset(trainset, train_inf_test_index)

    infset_loader = torch.utils.data.DataLoader(infset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    knownset_loader = torch.utils.data.DataLoader(knowntrainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=8)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    random.seed(18)
    testset_length = len(testset)
    testset_index = random.sample(range(testset_length), int(testset_length / 2))
    referenceset = torch.utils.data.Subset(testset, testset_index)
    test_inf_test_index = list(set(range(testset_length)).difference(set(testset_index)))
    test_infset = torch.utils.data.Subset(testset, test_inf_test_index)

    reference_loader = torch.utils.data.DataLoader(referenceset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8)
    test_infset_loader = torch.utils.data.DataLoader(test_infset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, test_loader, knownset_loader, infset_loader, reference_loader, test_infset_loader


def get_cifar10_dataloaders(args):

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # cifar10_mean=[0.485, 0.456, 0.406]
    # cifar10_std=[0.229, 0.224, 0.225]

    train_transform = DataAugmentation(args.img_size, args.img_size//2, cifar10_mean, cifar10_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=True)

    test_transform = DataAugmentation(args.img_size, args.img_size//2, cifar10_mean, cifar10_std, args.global_crops_scale, args.local_crops_scale,args.local_crops_number, is_train=False)

    trainset = datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=test_transform)

    trainset_length = len(trainset)
    random.seed(42)
    trainset_index = random.sample(range(trainset_length), int(trainset_length / 2))
    # unseen part to test inference model
    train_inf_test_index = list(set(range(trainset_length)).difference(set(trainset_index)))
    knowntrainset = torch.utils.data.Subset(trainset, trainset_index)
    infset = torch.utils.data.Subset(trainset, train_inf_test_index)

    infset_loader = torch.utils.data.DataLoader(infset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    knownset_loader = torch.utils.data.DataLoader(knowntrainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=8)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    random.seed(18)
    testset_length = len(testset)
    testset_index = random.sample(range(testset_length), int(testset_length / 2))
    referenceset = torch.utils.data.Subset(testset, testset_index)
    test_inf_test_index = list(set(range(testset_length)).difference(set(testset_index)))
    test_infset = torch.utils.data.Subset(testset, test_inf_test_index)

    reference_loader = torch.utils.data.DataLoader(referenceset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8)
    test_infset_loader = torch.utils.data.DataLoader(test_infset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, test_loader, knownset_loader, infset_loader, reference_loader, test_infset_loader



# if __name__=='__main__':
#     generate_gassian_ditribution(3,20)