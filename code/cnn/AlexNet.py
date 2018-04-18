#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AlexNet leNet 的升级版本
# AlexNet 网络相对于LeNet ，层数更深，同时第一次引入了激活层ReLU ，在全连接层引入了Dropout 层防止过拟合
# Alex Net 分七个模块
# https://upload-images.jianshu.io/upload_images/1689929-063fb60285b6ed42.png
# 模块1  2 简单的 卷积-激活函数-降采样-标准化
# 模块3 和4也是两个卷积过程，差别是少了降采样，原因就跟输入的尺寸有关，特征的数据量已经比较小了
# 模块五也是一个卷积过程，和模块一、二一样事儿的，就是重复重复。好了，
# 可以总结一下，模块一到五其实都是在做卷积运算，根据输入的图像尺寸在适当决定哪几层要用降采样。然后再加上一些必要的函数来控制数值
# 块六和七就是所谓的全连接层了，全连接层就和人工神经网络的结构一样的，结点数超级多，连接线也超多，所以这儿引出了一个dropout层，
# 来去除一部分没有足够激活的层，
# 模块八就是一个输出的结果，结合上softmax做出分类。有几类，输出几个结点，每个结点保存的是属于该类别的概率值
# 输入图片一般是:227*227


import torch
import torch.nn as nn
from torch.nn import  Sequential
class AlexNet(nn.Module):
    def __init__(self):
        super.__init__(AlexNet,self)
        num_classes = 1000
        layer1 = Sequential()
        layer1.add_module("conv-1", nn.Conv2d(3, 96, 11, stride=4))
        layer1.add_module("relu-1", nn.ReLU(inplace=True))
        layer1.add_module("pool-1", nn.MaxPool2d(3, stride=2))
        layer1.add_module("normal-1", nn.BatchNorm2d())
        self.layer1 = layer1
        layer2 = Sequential()
        layer2.add_module("conv-2", nn.Conv2d(96, 256, 5, padding=2, groups=2))
        layer2.add_module("relu-2", nn.ReLU(inplace=True))
        layer2.add_module("pool-2", nn.MaxPool2d(3, stride=2))
        layer2.add_module("normal-2", nn.BatchNorm2d())
        self.layer2 = layer2
        # 3 4 层没有下采样,因为图片已经很小了
        layer3 = Sequential()
        layer3.add_module("conv-3", nn.Conv2d(256, 384, 3, padding=1))
        layer3.add_module("relu-3", nn.ReLU(inplace=True))
        self.layer3 = layer3

        layer4 = Sequential()
        layer4.add_module("conv-4", nn.Conv2d(384, 384, 3, padding=1))
        layer4.add_module("relu-4", nn.ReLU(inplace=True))
        self.layer4 = layer4

        layer5 = Sequential()
        layer5.add_module("conv-5", nn.Conv2d(384, 256, 3, padding=2, groups=2))
        layer5.add_module("relu-5", nn.ReLU(inplace=True))
        layer5.add_module("pool-5", nn.MaxPool2d(3, stride=2))
        self.layer5 = layer5
        # 6 7 主要是全连接层
        layer6 = Sequential()
        layer6.add_module("full-6", nn.Linear(6*6*256, 4096))
        layer6.add_module("relu-6", nn.ReLU(inplace=True))
        layer6.add_module("drop-out-6,", nn.Dropout2d())

        self.layer6 = layer6
        layer7 = Sequential()
        layer7.add_module("full-7", nn.Linear(4096, num_classes))
        layer7.add_module("relu-7", nn.ReLU(inplace=True))
        layer7.add_module("drop-out-7,", nn.Dropout2d())
        self.layer7 = layer7

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(), 6*6*256)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    






