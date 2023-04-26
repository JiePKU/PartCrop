import torch
import torch.nn as nn


class Adversary(nn.Module):   #  black-box setting
    def __init__(self, input_dim=128):
        super(Adversary, self).__init__()
        self.input_dim = input_dim

        # for prediction
        self.pred_fc = nn.Sequential(nn.Linear(self.input_dim,1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,512),
                                     nn.ReLU(),
                                     nn.Linear(512,64),
                                     nn.ReLU(),
                                     nn.Linear(64,1))


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