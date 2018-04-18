#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""
    googlenet 2014年比赛冠军的model，这个model证明了一件事：
    用更多的卷积，更深的层次可以得到更好的结构。（当然，它并没有证明浅的层次不能达到这样的效果）
    下面不错的一篇文章关于 googleNet 的说明核心 可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能
    https://blog.csdn.net/shuzfan/article/details/50738394
    本实例只用来理解  GoogleNet里面的 Inception思想
"""
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x= nn.ReLU(x, inplace=True)
        x = F.relu(x, inplace=True)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branchPool = BasicConv2d(in_channels, pool_features, kernel_size=1)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branchPool(branch_pool, kernel_size=1)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs,1)


if __name__ == '__main__':
    model = Inception(1, 96)
    print(model)