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
parser.add_argument('--cp_path', type=str, help='check point path')
parser.add_argument('--data_path', default='../data', type=str, help='load find tune')
parser.add_argument('--save_path', type=str, help='check point path')
parser.add_argument('--model', type=str, help='check point path')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Data
# print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
transform_test = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Load fine tune model.
# print('==> find tune model..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
model = checkpoint['net']
if args.model == 'com':
    binary_predictor = SSDH_COM_VGG_RES_BINARY(model)
elif args.model == 'vgg':
    binary_predictor = SSDH_BINARY(model)
else:
    binary_predictor = SSDH_RES_BINARY(model)

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
pickle_file = open(args.save_path, 'wb')
pickle.dump(params, pickle_file)
pickle_file.close()
