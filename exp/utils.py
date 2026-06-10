# -*- coding: utf-8 -*-
"""可视化与保存指标"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_loss_curve(train_losses, val_accs, save_path):
    """绘制训练损失和验证准确率曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics(test_acc, classification_report_str, conf_matrix, save_path):
    """保存评估指标到文本文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("情感分析模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试集准确率 (Accuracy): {test_acc:.4f}\n\n")
        f.write("分类报告 (Classification Report):\n")
        f.write(classification_report_str + "\n")
        f.write("混淆矩阵 (Confusion Matrix):\n")
        f.write(np.array2string(conf_matrix, separator=', ') + "\n")