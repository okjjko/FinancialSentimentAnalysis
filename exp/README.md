# 金融新闻情感分析系统

基于双向LSTM的金融情感分析系统，支持文本情感分类和Gradio交互界面。

## 数据集

来自kaggle网站的Emotions dataset for NLP
地址：[Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data)

## 环境配置

```
pip install torch==2.0.1
pip install gradio==4.26.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install numpy==1.24.3
```

## 项目结构

```
exp/
│
├── config.py                # 全局配置参数（超参数、路径、输出目录）
├── data_loader.py           # 数据集类、词汇表构建、数据预处理
├── model.py                 # BiLSTM 模型定义
├── train.py                 # 训练脚本
├── inference.py             # 模型加载与推理函数
├── app.py                   # Gradio Web 界面
├── utils.py                 # 可视化辅助函数（损失曲线、混淆矩阵、保存指标）
│
├── data/                    # 数据目录
│   ├── train.txt            # 训练集，每行：文本;标签
│   ├── val.txt              # 验证集
│   └── test.txt             # 测试集
│
└── output/                  # 自动生成，存放所有输出文件
    ├── best_model.pth       # 最佳模型权重
    ├── vocab.pth            # 词汇表
    ├── label_map.pth        # 标签映射
    ├── loss_curve.png       # 训练损失与验证准确率曲线图
    ├── confusion_matrix.png # 混淆矩阵热力图
    └── metrics.txt          # 评估报告（准确率、分类报告、混淆矩阵）
```

## 运行说明

```
python train.py // 训练模型

python app.py // 启动 Web 应用
```

