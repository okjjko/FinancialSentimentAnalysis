# -*- coding: utf-8 -*-
"""BiLSTM 模型定义与 Focal Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # 拼接双向最后一层
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss，用于处理难分类样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha      # 类别权重 tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss