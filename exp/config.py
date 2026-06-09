# -*- coding: utf-8 -*-
"""全局配置文件"""

import torch
import os

# 模型超参数
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0005
WEIGHT_DECAY = 1e-4
MAX_LEN = 64
PATIENCE = 5
GRAD_CLIP = 1.0
DROPOUT = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 文件路径
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
VAL_FILE = os.path.join(DATA_DIR, 'val.txt')
TEST_FILE = os.path.join(DATA_DIR, 'test.txt')

# 输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
VOCAB_PATH = os.path.join(OUTPUT_DIR, 'vocab.pth')
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, 'label_map.pth')

# 评估结果保存路径
LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, 'loss_curve.png')
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.txt')