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

class SSDH_COM_VGG_RES(nn.Module):
    def __init__(self, vgg, res, H):
        super(SSDH_COM_VGG_RES, self).__init__()
        # vgg net
        self.features = nn.Sequential(*list(vgg.features.children()))
        self.f2h_vgg = nn.Linear(512, H)
        # res net
        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        self.f2h_res = nn.Linear(512, H)
        # vgg + res
        self.h2o = nn.Linear(H, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # vgg net
        out = self.features(x)
        out_vgg = out.view(out.size(0), -1)
        # res net
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out_res = out.view(out.size(0), -1)
        # together
        hidden = self.sigmoid(self.f2h_vgg(out_vgg) + self.f2h_res(out_res))
        output = self.h2o(hidden)
        return hidden, output

class SSDH_COM_VGG_RES_BINARY(nn.Module):
    def __init__(self, com):
        super(SSDH_COM_VGG_RES_BINARY, self).__init__()
        # vgg net
        self.features = com.features
        self.f2h_vgg = com.f2h_vgg
        # res net
        self.conv1 = com.conv1
        self.bn1 = com.bn1
        self.layer1 = com.layer1
        self.layer2 = com.layer2
        self.layer3 = com.layer3
        self.layer4 = com.layer4
        self.f2h_res = com.f2h_res
        # together
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # vgg net
        out = self.features(x)
        out_vgg = out.view(out.size(0), -1)
        # res net
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out_res = out.view(out.size(0), -1)
        # together
        hidden = self.sigmoid(self.f2h_vgg(out_vgg) + self.f2h_res(out_res))
        hidden = (torch.sign(hidden - 0.5) + 1) / 2
        return hidden