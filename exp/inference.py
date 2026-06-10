# -*- coding: utf-8 -*-
"""模型加载与推理"""

import re
import torch
from config import DEVICE, MAX_LEN, MODEL_PATH, VOCAB_PATH, LABEL_MAP_PATH
from data_loader import clean_text
from model import BiLSTMClassifier


def load_model_and_vocab():
    vocab = torch.load(VOCAB_PATH)
    label2id = torch.load(LABEL_MAP_PATH)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    model = BiLSTMClassifier(len(vocab), num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, vocab, id2label


def preprocess_text(text, vocab, max_len=MAX_LEN):
    text = clean_text(text)          # 与训练时完全一致
    words = text.split()
    ids = [vocab.get(w, vocab.get('<unk>', 1)) for w in words[:max_len]]
    if len(ids) < max_len:
        ids += [vocab.get('<pad>', 0)] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def predict(text, model, vocab, id2label):
    input_ids = preprocess_text(text, vocab).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}