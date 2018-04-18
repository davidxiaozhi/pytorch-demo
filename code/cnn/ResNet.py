#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# ResNet(Deep Residual Learning) 2015 年竞赛的冠军,微软出品,通过残差模块能够训练高到152层次的网络
# 其设计灵感来源于:在不断加深神经网络时,会出现一个反degradation,即准确率会先上升然后达到饱和,在持续加深深度则会导致模型准确率下降
# 这个不是过拟合的问题,因为不仅仅在验证集上误差增加,在训练集上误差也会增加
# 假设 一个比较浅层的网络 准确率达到饱和,在后面添加几个恒等映射层,误差不会增加,也就说更深的模型不会导致模型的效果下降
# 恒等映射 直接将前一层的输出传递到后面的思想就是 Resnet 得了灵感来源,
# 假设每某个神经网络 输入 x ,期望输出 H(x) .如果直接把输入 x 传递到输出作为结果, n那么要学习的目标就变成了F(x) = H(x) - x
# 本实例只用来理解 ResNet 当中的残差模块
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes , stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #3*3 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

