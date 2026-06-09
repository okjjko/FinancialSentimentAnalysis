"""FastAPI 启动入口"""

import os
import sys
from pathlib import Path

import uvicorn

# 切换到 exp/ 目录以兼容原有相对路径配置
_exp_dir = Path(__file__).resolve().parent / "exp"
os.chdir(str(_exp_dir))
sys.path.insert(0, str(_exp_dir))

# 此时 CWD = exp/，config.py 中的相对路径 'output/' 等可正确解析
from api import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Financial Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
