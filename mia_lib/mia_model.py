import torch
import torch.nn as nn


"""
This model is referred to paper:

"Machine Learning with Membership Privacy using Adversarial Regularization"

More detail can be found in:
https://dl.acm.org/doi/abs/10.1145/3243734.3243855

this code is implemented in 2022.01.03 

version : v1

"""

class Adversary(nn.Module):   #  black-box setting
    def __init__(self, input_dim=128, attacker_type='default'):
        super(Adversary, self).__init__()
        self.input_dim = input_dim
        
        if attacker_type=='default':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,512),
                                     nn.ReLU(),
                                     nn.Linear(512,256),
                                     nn.ReLU(),
                                     nn.Linear(256,128),
                                     nn.ReLU(),
                                     nn.Linear(128,1))
        
        elif attacker_type=='wide':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,512),
                                     nn.ReLU(),
                                     nn.Linear(512,256),
                                     nn.ReLU(),
                                     nn.Linear(256,1))
        
        elif attacker_type=='narrow':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,256),
                                     nn.ReLU(),
                                     nn.Linear(256,128),
                                     nn.ReLU(),
                                     nn.Linear(128,64),
                                     nn.ReLU(),
                                     nn.Linear(64,1))
        
        elif attacker_type == 'deep':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,512),
                                     nn.ReLU(),
                                     nn.Linear(512,256),
                                     nn.ReLU(),
                                     nn.Linear(256,256), ## add
                                     nn.ReLU(),
                                     nn.Linear(256,128),
                                     nn.ReLU(),
                                     nn.Linear(128,1))

        elif attacker_type == 'shallow':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,512),
                                     nn.ReLU(),
                                     nn.Linear(512,128),
                                     nn.ReLU(),
                                     nn.Linear(128,1))


        # init weight
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self,x):
        # x should be softmax or sigmod output

        out = self.pred_fc(x) # B C
        out = torch.sigmoid(out)
        return out

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            if m.bias.data is not None:
                m.bias.data.fill_(0)


"""
version : v2

Two main modification for stablizing training:
    1. replace ReLU with Tanh()
    2. introduce normalization to scale the input

""" 

class RMSNorm(nn.Module):
    def __init__(self, dim=128):
        super(RMSNorm, self).__init__()
        self.dim = dim

    def forward(self, x):

        return x / torch.sqrt((x**2).mean(dim=1, keepdim=True)+1e-6)


class Adversary_v2(nn.Module):   #  black-box setting
    def __init__(self, input_dim=128, attacker_type='default'):
        super(Adversary, self).__init__()
        self.input_dim = input_dim
        
        self.norm = RMSNorm(input_dim)
        
        if attacker_type=='default':
            self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,512),
                                     RMSNorm(512),
                                     nn.Tanh(),
                                     nn.Linear(512,256),
                                     RMSNorm(256),
                                     nn.Tanh(),
                                     nn.Linear(256,128),
                                     RMSNorm(128),
                                     nn.Tanh(),
                                     nn.Linear(128,1))

        # init weight
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self,x):


        out = self.pred_fc(x) # B C
        out = torch.sigmoid(out)
        return out

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            if m.bias.data is not None:
                m.bias.data.fill_(0)
