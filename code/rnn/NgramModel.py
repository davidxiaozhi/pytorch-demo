#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class NGramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, n_dim)
        self.linear1 = nn.Linear(context_size*n_dim,128)
        self.linear2 = nn.Linear(128, self.vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        log_prob = F.log_softmax(out)
        return log_prob
