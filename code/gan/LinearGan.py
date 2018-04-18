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

class discriminator(nn.Module):
    """
        图片都是 28 * 28 负责判定图片真假
    """
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784*256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x;



class genearator(nn.Module):
    """生成图片"""
    def __init__(self, input_size):
        super(genearator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            # 使用 tanh 将数据规范化为 -1 到 1之前 这是应为输入的图片数据会规范化到 -1 到 1 之间
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)



D = discriminator()
G = genearator()
#定义损失函数 二分类的损失函数
criterion = nn.BCELoss()
#定义优化器
d_optmizer = torch.optim.Adam(D.parameters(), lr=0.003)
g_optmizer = torch.optim.Adam(G.parameters(), lr=0.003)

# 下面进行训练生成器和判定器









