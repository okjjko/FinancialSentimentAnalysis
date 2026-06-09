# -*- coding: utf-8 -*-
"""数据加载与预处理"""

import re
import torch
from torch.utils.data import Dataset
from collections import Counter
from config import MAX_LEN

class EmotionDataset(Dataset):
    def __init__(self, file_path, vocab=None, label2id=None, max_len=MAX_LEN, build_vocab=False):
        self.max_len = max_len
        self.samples = []
        self.labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) != 2:
                    continue
                text, label = parts[0], parts[1]
                # 清洗：只保留字母和空格，转小写
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                words = text.split()
                if len(words) == 0:
                    continue
                self.samples.append(words)
                self.labels.append(label)
        
        if build_vocab:
            all_labels = list(set(self.labels))
            self.label2id = {lab: idx for idx, lab in enumerate(all_labels)}
            self.id2label = {idx: lab for lab, idx in self.label2id.items()}
        else:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
        
        if build_vocab:
            word_counts = Counter()
            for words in self.samples:
                word_counts.update(words)
            self.vocab = {'<pad>': 0, '<unk>': 1}
            for word, _ in word_counts.items():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab
        
        self.pad_idx = self.vocab.get('<pad>', 0)
        self.unk_idx = self.vocab.get('<unk>', 1)
    
    def encode(self, words):
        ids = [self.vocab.get(w, self.unk_idx) for w in words[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids = self.encode(self.samples[idx])
        label = self.label2id[self.labels[idx]]
        return input_ids, torch.tensor(label, dtype=torch.long)

def build_vocab_from_dataset(file_path):
    """辅助函数：从文件构建词汇表和标签映射（用于快速查看）"""
    dataset = EmotionDataset(file_path, build_vocab=True)
    return dataset.vocab, dataset.label2id