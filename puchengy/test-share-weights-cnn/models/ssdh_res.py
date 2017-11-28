'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SSDH_RES(nn.Module):
    def __init__(self, res, H):
        super(SSDH_RES, self).__init__()
        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        self.f2h = nn.Linear(512, H)
        self.h2o = nn.Linear(H, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        hidden = self.sigmoid(self.f2h(out))
        output = self.h2o(hidden)
        return hidden, output