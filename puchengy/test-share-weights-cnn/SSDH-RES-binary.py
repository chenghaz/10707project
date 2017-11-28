'''
    SSDH-VGG main function
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from torch.autograd import Variable
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='PyTorch SSDH-binary Prediction')
parser.add_argument('--cp_path', default='./checkpoint/ssdh', type=str, help='load find tune')
parser.add_argument('--data_path', default='./data', type=str, help='load find tune')
parser.add_argument('--K', default=64, type=int, help='hidden units number')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load fine tune model.
print('==> find tune model..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
ssdh = checkpoint['net']
binary_predictor = SSDH_BINARY(ssdh)

# cuda usage
if use_cuda:
    binary_predictor.cuda()
    binary_predictor = torch.nn.DataParallel(binary_predictor, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# test data
flag = 0
binary_code_test = None
def test():
    binary_predictor.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        hidden = binary_predictor(inputs)
        hidden = hidden.cpu().data.numpy()
        targets = targets.cpu().data.numpy()
        result = np.concatenate((hidden, np.reshape(targets, (targets.size, 1))), 1)
        global binary_code_test
        global flag
        if flag == 0:
            binary_code_test = result
            flag = 1
        else:
            binary_code_test = np.concatenate((binary_code_test, result))
test()

# train data
flag = 0
binary_code_train = None
def train():
    binary_predictor.eval()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        hidden = binary_predictor(inputs)
        hidden = hidden.cpu().data.numpy()
        targets = targets.cpu().data.numpy()
        result = np.concatenate((hidden, np.reshape(targets, (targets.size, 1))), 1)
        global binary_code_train
        global flag
        if flag == 0:
            binary_code_train = result
            flag = 1
        else:
            binary_code_train = np.concatenate((binary_code_train, result))
train()

params = {
    'binary_code_test': binary_code_test,
    'binary_code_train': binary_code_train
}
pickle_file = open('./checkpoint/binary_code_' + str(args.K) + '_1_loss', 'wb')
pickle.dump(params, pickle_file)
pickle_file.close()
