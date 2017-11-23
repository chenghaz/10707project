''''
    SSDH-VGG
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

class SSDH(nn.Module):
    def __init__(self, vgg, H):
        super(SSDH, self).__init__()
        self.features = nn.Sequential(*list(vgg.features.children()))
        self.f2h = nn.Linear(512, H)
        self.h2o = nn.Linear(H, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        hidden = self.sigmoid(self.f2h(out))
        output = self.h2o(hidden)
        return hidden, output

class SSDH_BINARY(nn.Module):
    def __init__(self, vgg):
        super(SSDH_BINARY, self).__init__()
        self.features = nn.Sequential(*list(vgg.features.children()))
        self.f2h = vgg.f2h
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        hidden = (torch.sign(self.sigmoid(self.f2h(out)) - 0.5) + 1) / 2
        return hidden