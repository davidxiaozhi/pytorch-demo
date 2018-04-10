#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 使用 pytorch 进行手写体识别 官方例子
#
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.jupyter)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#定义 code 当中的随机因子
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#加载训练数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
#加载测试数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        前向反馈神经网络构建
        有关 pytorch 的卷积神经网络介绍 http://pytorch.org/docs/master/nn.html
        有关 pytorch 的 最大池化卷积介绍 http://pytorch.org/docs/master/nn.html
        卷积 conv2d 输入输出 第一参数为批次  第二参数为输入渠道  第三,四参数为 高和宽
        Input: (N,Cin,Hin,Win)
        Output: (N,Cout,Hout,Wout) where
        卷积后的高
        Hout=( (Hin+2∗padding[0]−dilation[0]∗(kernel_size[0]−1) −1) / stride[0] )+1

        Wout=( (Win+2∗padding[1]−dilation[1]∗(kernel_size[1]−1) −1) / stride[1]) +1
        输入为 [batch,channel,height,width](64,1,28,28)

        默认参数 padding:0 其他为1
        第一 卷积 为(本图片宽高一致) ( ( (28+2*0 - 1*(5-1)) - 1 ) / 1 ) +1 = 24
        最大池化 2-max 池化  ( ( (24+2*0 - 1*(5-1)) - 1 ) / 1 ) +1 = 23 池化后 12
        第二 卷积 ( (12+2*0-1*(5-1) -1)/1 )+1 = 8 池化后为 4 即 (64,20,4,4)
        下面我们进行全连接 将 tensor 进行矩阵变化 变成 64 * 320 即 320是 channel*height*width 这样批次不受影响
        第三 进行第一次全连接 320*50  结果 64* 50
        第四 进行第二次全连接 50*10   64*10
        第五 softmax 结果 64
        :param x: 输入为 [batch,channel,height,width](64,1,28,28)
        :return:
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #channel 由 1 变化为10
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    print(model)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        ##默认执行的是 model.__call__内部函数,其内部调用 forward
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval() #model.eval就可以进行测试了
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()