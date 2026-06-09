# -*- coding: utf-8 -*-
"""模型定义"""

import torch
import torch.nn as nn
from config import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT,
                            bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        # 取最后一层双向的 hidden state 并拼接
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits