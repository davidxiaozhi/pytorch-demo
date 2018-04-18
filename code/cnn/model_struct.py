#!/usr/bin/env python
# -*- coding:UTF-8 -*-
#分析 pytorch 构建的网络的结构
import torch
from code.cnn.minist_cnn import Net

model = Net()
# 打印模型本身的网络结构
print(model)
"""
children()直接取得是网络结构的当前层的下一层的结构
"""
new_model = torch.nn.Sequential(*list(model.children())[:2])
print(new_model)
"""
modules 不仅可以返回所有的迭代器这样的好处,可以遍历访问,所有的网络结构

每一层 有一个与它们相对应的是named_children ()属性
以及named_modules () ， 这两个不仅会返回模块的迭代器，还会返问网络层的名字3

使用isinstance 可以判断这个模块是不是所需要的类型实例，这样就提取出了
所有的卷积模块，
"""
conv_model = torch.nn.Sequential()
for layer in model.named_modules () :
    if isinstance(layer[1], torch.nn.Conv2d):
        conv_model.add_module(layer[0], layer[1])
print(conv_model)

#parameters 是返回所有的参数 而 named_parameters是返回参数的名字和跌打器
for param in model.parameters():
    print(param[0])

for name, p in model.named_parameters():
   #print("name:{} Parameters:{}".format(name, p))
    print("name:{}".format(name))

