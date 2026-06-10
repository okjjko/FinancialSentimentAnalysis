# -*- coding: utf-8 -*-
"""全局配置参数"""

import torch
import os

# 模型超参数
EMBED_DIM = 200
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0003
WEIGHT_DECAY = 5e-5
MAX_LEN = 80
PATIENCE = 7
GRAD_CLIP = 1.0
DROPOUT = 0.4

# 类别权重自定义系数（针对训练集中不均衡或难分情绪）
CLASS_WEIGHT_FACTOR = {
    'anger': 2.0,
    'surprise': 1.5,
    'fear': 1.2,
    # 其他类别保持默认 1.0
}

# 数据增强概率
AUGMENT_PROB = 0.3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 文件路径
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
VAL_FILE = os.path.join(DATA_DIR, 'val.txt')
TEST_FILE = os.path.join(DATA_DIR, 'test.txt')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
VOCAB_PATH = os.path.join(OUTPUT_DIR, 'vocab.pth')
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, 'label_map.pth')
LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.txt')