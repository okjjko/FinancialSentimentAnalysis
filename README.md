# 金融新闻情感分析系统

基于双向 LSTM 的文本情感分析系统，支持 6 类情感分类。前端基于 React + FastAPI 构建。

## 数据集

来自 Kaggle：[Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data)

- 6 类情感：sadness, surprise, love, joy, anger, fear
- 训练集 16000 条，验证集 2000 条，测试集 2000 条

## 环境要求

- **Python**: 3.9+
- **Node.js**: 18+（用于前端开发服务）

## 环境配置

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装前端依赖
cd web && npm install
```

## 项目结构

```
FinancialSentimentAnalysis/
├── exp/
│   ├── api.py                  # FastAPI 路由（预测、历史、示例等）
│   ├── config.py               # 全局配置
│   ├── data_loader.py          # 数据集与预处理
│   ├── model.py                # BiLSTM 模型定义
│   ├── train.py                # 训练脚本
│   ├── inference.py            # 模型加载与推理
│   ├── utils.py                # 可视化工具
│   ├── data/                   # 数据目录
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   └── output/                 # 模型输出
│       ├── best_model.pth
│       ├── vocab.pth
│       └── label_map.pth
├── web/                        # React 前端
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       ├── api.js
│       └── components/
│           ├── Header.jsx
│           ├── Sidebar.jsx
│           ├── SingleTab.jsx
│           ├── BatchTab.jsx
│           ├── DetailPanel.jsx
│           ├── ResultLabel.jsx
│           └── Footer.jsx
├── run_api.py                  # FastAPI 启动入口
├── requirements.txt
└── README.md
```

## 训练模型

```bash
# 进入 exp/ 目录运行
cd exp && python train.py
```

训练完成后，模型文件保存在 `exp/output/`。

## 启动 Web 应用

需要同时启动后端和前端两个服务：

```bash
# 终端 1 — 后端 API（http://localhost:8000）
python run_api.py

# 终端 2 — 前端开发服务器（http://localhost:3000）
cd web && npm run dev
```

浏览器打开 `http://localhost:3000` 即可使用。

## API 接口

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/status` | 模型状态与类别信息 |
| GET | `/api/examples` | 示例文本列表 |
| POST | `/api/predict/single` | 单条文本情感分析 |
| POST | `/api/predict/batch` | 上传 .txt 文件批量分析 |
| GET | `/api/history` | 分析历史记录列表 |
| GET | `/api/history/{id}` | 单条历史详情 |
| DELETE | `/api/history` | 清空历史 |

## 模型架构

- **模型**: BiLSTM（双向 2 层 LSTM）
- **嵌入层**: 256 维
- **隐藏层**: 256 维
- **Dropout**: 0.5
- **优化器**: Adam (lr=5e-4, weight_decay=1e-4)
- **序列长度**: 64

## 结果

| 指标 | 值 |
|------|-----|
| 测试准确率 | 88.80% |
| 加权平均 F1 | 0.8871 |
| Macro F1 | 0.8383 |
