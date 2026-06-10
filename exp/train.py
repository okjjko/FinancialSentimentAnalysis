# -*- coding: utf-8 -*-
"""训练脚本：数据加载、模型训练、评估与保存"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from data_loader import EmotionDataset
from model import BiLSTMClassifier, FocalLoss
from utils import plot_loss_curve, plot_confusion_matrix, save_metrics


class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc, model, vocab, label2id, epoch):
        if self.best_score is None:
            self.best_score = val_acc
            self.save_checkpoint(model, vocab, label2id, epoch)
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.save_checkpoint(model, vocab, label2id, epoch)
            self.counter = 0

    def save_checkpoint(self, model, vocab, label2id, epoch):
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(vocab, VOCAB_PATH)
        torch.save(label2id, LABEL_MAP_PATH)
        print(f"  [Epoch {epoch+1}] 保存最佳模型 (val_acc={self.best_score:.4f})")


def train_model():
    print("加载训练数据...")
    train_dataset = EmotionDataset(TRAIN_FILE, build_vocab=True, train=True, augment=True)
    vocab = train_dataset.vocab
    label2id = train_dataset.label2id
    id2label = {v: k for k, v in label2id.items()}
    print(f"词汇表大小: {len(vocab)}")
    print(f"标签映射: {label2id}")

    # 类别权重 + 自定义系数
    label_counts = Counter(train_dataset.labels)
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    base_weights = {lab: total / (n_classes * count) for lab, count in label_counts.items()}
    final_weights = {}
    for lab in base_weights:
        factor = CLASS_WEIGHT_FACTOR.get(lab, 1.0)
        final_weights[lab] = base_weights[lab] * factor
    weights_tensor = torch.tensor([final_weights[lab] for lab in sorted(label2id.keys())],
                                  dtype=torch.float).to(DEVICE)
    print(f"类别权重（调整后）: {dict(zip(sorted(label2id.keys()), weights_tensor.cpu().numpy()))}")

    # 加载验证/测试集
    val_dataset = EmotionDataset(VAL_FILE, vocab=vocab, label2id=label2id, train=False)
    test_dataset = EmotionDataset(TEST_FILE, vocab=vocab, label2id=label2id, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BiLSTMClassifier(len(vocab), len(label2id)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)

    early_stopping = EarlyStopping()
    train_losses = []
    val_accs = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)
        early_stopping(val_acc, model, vocab, label2id, epoch)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    # 绘制损失/准确率曲线
    plot_loss_curve(train_losses, val_accs, LOSS_CURVE_PATH)

    # 加载最佳模型并全面评估测试集
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"测试集准确率: {test_acc:.4f}")

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\n分类报告:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, target_names, CONFUSION_MATRIX_PATH)
    save_metrics(test_acc, report, cm, METRICS_PATH)

    return model, vocab, label2id


if __name__ == '__main__':
    train_model()