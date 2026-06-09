# -*- coding: utf-8 -*-
"""训练、验证、测试、评估及可视化"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from data_loader import EmotionDataset
from model import BiLSTMClassifier
from utils import plot_loss_curve, plot_confusion_matrix, save_metrics

class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc, model, vocab, label2id, epoch):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, vocab, label2id, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, vocab, label2id, epoch)
            self.counter = 0
    
    def save_checkpoint(self, model, vocab, label2id, epoch):
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(vocab, VOCAB_PATH)
        torch.save(label2id, LABEL_MAP_PATH)
        print(f"  [Epoch {epoch+1}] 保存最佳模型 (val_acc={self.best_score:.4f})")

def train_model():
    print("加载训练数据...")
    train_dataset = EmotionDataset(TRAIN_FILE, build_vocab=True)
    vocab = train_dataset.vocab
    label2id = train_dataset.label2id
    id2label = {v: k for k, v in label2id.items()}
    print(f"词汇表大小: {len(vocab)}")
    print(f"标签映射: {label2id}")
    
    # 类别权重
    label_counts = Counter(train_dataset.labels)
    total = sum(label_counts.values())
    class_weights = {lab: total / (len(label_counts) * count) for lab, count in label_counts.items()}
    weights_tensor = torch.tensor([class_weights[label] for label in sorted(label2id.keys())], dtype=torch.float).to(DEVICE)
    print(f"类别权重: {weights_tensor}")
    
    val_dataset = EmotionDataset(VAL_FILE, vocab=vocab, label2id=label2id)
    test_dataset = EmotionDataset(TEST_FILE, vocab=vocab, label2id=label2id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = BiLSTMClassifier(len(vocab), len(label2id)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    early_stopping = EarlyStopping()
    
    # 记录损失和准确率
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
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_accs, LOSS_CURVE_PATH)
    
    # 加载最佳模型并在测试集上全面评估
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
    
    # 计算准确率
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 分类报告
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\n分类报告:\n", report)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, target_names, CONFUSION_MATRIX_PATH)
    
    # 保存所有指标到文件
    save_metrics(test_acc, report, cm, METRICS_PATH)
    
    return model, vocab, label2id

if __name__ == '__main__':
    train_model()