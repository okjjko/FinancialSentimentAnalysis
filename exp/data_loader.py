# -*- coding: utf-8 -*-
"""数据加载、预处理与增强"""

import re
import random
import torch
from torch.utils.data import Dataset
from collections import Counter

from config import MAX_LEN, AUGMENT_PROB

# 尝试导入 nltk 用于同义词替换（若未安装则跳过增强）
try:
    import nltk
    from nltk.corpus import wordnet
    nltk.data.find('corpora/wordnet')
except (LookupError, ImportError):
    wordnet = None
    print("Warning: NLTK WordNet not available, data augmentation disabled.")


def clean_text(text):
    """
    清洗文本：保留字母、空格、感叹号、问号，转为小写。
    连续标点（!!!、???）简化为单个。
    """
    text = text.lower()
    # 保留字母、空格、! 和 ?
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    # 将连续的 ! 或 ? 替换为单个
    text = re.sub(r'[!?]+', lambda m: m.group(0)[0], text)
    return text


def synonym_replace(words, n=1):
    """同义词替换增强，随机替换 n 个词"""
    if wordnet is None:
        return words
    new_words = words.copy()
    indices = list(range(len(new_words)))
    random.shuffle(indices)
    replaced = 0
    for idx in indices:
        word = new_words[idx]
        synsets = wordnet.synsets(word)
        if synsets:
            # 取第一个同义词的第一个词（词形可能不同）
            replacement = synsets[0].lemmas()[0].name().replace('_', ' ')
            # 只替换单个词，不考虑多词
            if ' ' not in replacement:
                new_words[idx] = replacement
                replaced += 1
                if replaced >= n:
                    break
    return new_words


class EmotionDataset(Dataset):
    def __init__(self, file_path, vocab=None, label2id=None, max_len=MAX_LEN,
                 build_vocab=False, train=False, augment=False):
        self.max_len = max_len
        self.train = train
        self.augment = augment and train  # 仅训练集可增强
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
                text = clean_text(text)
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
        words = self.samples[idx]
        # 数据增强（随机同义词替换）
        if self.augment and random.random() < AUGMENT_PROB:
            words = synonym_replace(words, n=1)
        input_ids = self.encode(words)
        label = self.label2id[self.labels[idx]]
        return input_ids, torch.tensor(label, dtype=torch.long)