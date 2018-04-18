#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from torch.autograd import Variable

training_data =[
    ("The dog ate the apple". split(), ["DET" , "NN" , "V" , "DET" , "NN"]),
    ("Everybody read that book" .split(), ["NN" , "V" , "DET" , "NN"])
]

word_to_idx = {}
tag_to_idx = {}

for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)

alphabet = 'abcdefghijklmnopqrstuvwχyz'
character_to_idx = {}
for i in range(len(alphabet) ):
    character_to_idx[alphabet[i]] = i

class CharLSTM(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char,char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)


    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[0]


class LSTMTagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim+char_hidden, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, n_tag)


    def forward(self, x, word_data):
        word = [i  for i in word_data]
        char = torch.FloatTensor()
        for each in word:
            word_list = []
            for letter in each:
                word_list.append(character_to_idx[letter.lower()])
            word_list = torch.LongTensor(word_list)
            word_list = word_list.unsqueeze(0)
            temp_char = self.char_lstm(Variable(word_list))
            temp_char = temp_char.squeeze (0)
            char = torch.cat((char, temp_char.data), 0)
        char = char.squeeze(1)
        char = Variable(char)
        x = self.word_embedding(x)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear1(x)
        y = F.softmax(x)
        return y





