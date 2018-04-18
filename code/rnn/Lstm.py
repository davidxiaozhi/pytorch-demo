#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""
    LSTM  长短期记忆模型,我们先简单介绍一下其原理机制
    历史记忆衰减: 首先将历史网络输出 h(t-1) 和 当前时刻的输入 x(t) 结合起来 然后做线性变化 f(t) σ * {W(f) * [ h(t-1), x(t) ] + b(f)}

    t时刻学到的记忆是如何计算的,
    当前时刻记忆衰减系数 i(t) σ * {W(i) * [ h(t-1), x(t) ] + b(i)}
    当前时刻学到记忆存储 C'(t) = tanh{W(c) * [ h(t-1), x(t) ] + b(c)}
    当前时刻的记忆状态  C(t) = f(t)*(C(t-1) 十i(t)*C'(t)

    输出
        输出系数 o(t) σ * {W(o) * [ h(t-1), x(t) ] + b(o)}
        输出结果 h(t) = O(t)* tanh(C(t))

    RNN 的 loss 跳跃问题

    跳跃的loss不收敛: 基于循环的神经网络就会出现这样的情况- 在随后的研究中，人们发现出现这种情况的根本原因是因为RNN 的误差曲面粗糙不平
    是梯度裁剪( gradient clipping ) : 使用梯度裁剪能够将大的梯度裁掉，这样就能够在一定程度上避免收敛不好的问题

"""
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

class RNN_LSTM(nn.Module):
    """
    LSTM batchFirst = True
       output (batch ,seq, hidden * direction)
       h(记忆单元) (lαyer * dírecti例， batch , hidden) ,
    """
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNN_LSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 网络的输出是(batch, seq, hidden * diredion)  batchFirst=True 状态输出 状态 (batch, layer*direction, hidden)
        #
        #，这是因为循环神经网络的输出也是一个序列，这一行代码是取出输出序列中的最后一个
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    lstm = nn.LSTM(input_size=20, hidden_size=50,num_layers=2)
    print(lstm.all_weights)