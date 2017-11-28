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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--K', default=64, type=int, help='hidden layer length')
parser.add_argument('--data_path', default='./data', type=str, help='data_path')
parser.add_argument('--cp_path', default='./checkpoint/Res18ckpt.t7', type=str, help='check point path')
parser.add_argument('--progress', default=True, type=bool, help='show progress bar')
args = parser.parse_args()

progress = True
if progress:
    from utils import progress_bar

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
print('==> find tunine model..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
net = checkpoint['net']
ssdh = SSDH_RES(net, args.K)

# cuda usage
if use_cuda:
    ssdh.cuda()
    ssdh = torch.nn.DataParallel(ssdh, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def loss_function2(hidden):
    return Variable.sum(Variable.mean(Variable.pow(Variable.add(hidden, -0.5), 2), 1))

def loss_function3(hidden):
    return Variable.sum(Variable.pow(Variable.add(Variable.mean(hidden, 1), -0.5), 2))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ssdh.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        hidden, outputs = ssdh(inputs)
        # loss1: cross entropy
        loss1 = criterion(outputs, targets)
        # loss2: force to 0/1
        # loss2 = loss_function2(hidden)
        # loss3: force 50%, 50%
        # loss3 = loss_function3(hidden)
        # loss = loss1 - loss2 + loss3
        loss = loss1
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if progress:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    ssdh.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        hidden, outputs = ssdh(inputs)
        # loss1: cross entropy
        loss1 = criterion(outputs, targets)
        # loss2: force to 0/1
        # loss2 = loss_function2(hidden)
        # loss3: force 50%, 50%
        # loss3 = loss_function3(hidden)
        loss = loss1

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if progress:
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': ssdh.module if use_cuda else ssdh,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ssdh_res_' + str(args.K) + '_1_loss')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
