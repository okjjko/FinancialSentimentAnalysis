"""FastAPI 路由 — 金融情感分析系统 API"""

import os
import random
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

# 确保 exp/ 在 sys.path 中以兼容原有导入方式
_exp_root = str(Path(__file__).resolve().parent)
if _exp_root not in sys.path:
    sys.path.insert(0, _exp_root)

from config import MODEL_PATH, VOCAB_PATH, LABEL_MAP_PATH
from inference import load_model_and_vocab, predict as predict_text

router = APIRouter(prefix="/api")

_model = None
_vocab = None
_id2label = None
_model_lock = threading.Lock()

_history = []
_history_counter = 0
_history_lock = threading.Lock()
_HISTORY_MAX = 200

DATA_FILE = Path(__file__).resolve().parent / "data" / "test.txt"


def _get_model():
    global _model, _vocab, _id2label
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError("模型文件不存在，请先训练")
                _model, _vocab, _id2label = load_model_and_vocab()
    return _model, _vocab, _id2label


def _add_history(text, results, source):
    global _history, _history_counter
    with _history_lock:
        _history_counter += 1
        entry = {
            "id": _history_counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "source": source,
            "results": results,
        }
        _history.append(entry)
        if len(_history) > _HISTORY_MAX:
            _history.pop(0)
        return entry


# ─── 状态 ─────────────────────────────────────────

@router.get("/status")
def get_status():
    label_names = sorted([
        "sadness", "surprise", "love", "joy", "anger", "fear"
    ])
    return {
        "model": "BiLSTM",
        "classes": 6,
        "labels": label_names,
        "status": "ok",
    }


# ─── 示例文本 ─────────────────────────────────────

@router.get("/examples")
def get_examples():
    if not DATA_FILE.exists():
        return {"examples": []}
    try:
        lines = DATA_FILE.read_text(encoding="utf-8").splitlines()
        random.shuffle(lines)
        examples = []
        for line in lines[:6]:
            if ";" in line:
                text = line.split(";")[0].strip()
                examples.append({"text": text})
        return {"examples": examples}
    except Exception:
        return {"examples": []}


# ─── 文本预测 ────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

@router.post("/predict/single")
async def predict_single(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "请输入文本")
    try:
        model, vocab, id2label = _get_model()
        raw = predict_text(req.text, model, vocab, id2label)
        results = [{"class": k, "confidence": v} for k, v in raw.items()]
        results.sort(key=lambda x: x["confidence"], reverse=True)
        _add_history(req.text, results, "单条")
        return {"text": req.text, "results": results}
    except Exception as e:
        raise HTTPException(500, f"推理失败: {e}")


# ─── 批量预测 ────────────────────────────────────

@router.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(400, "请上传文本文件")
    try:
        content = await file.read()
        text = content.decode("utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            raise HTTPException(400, "文件为空")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件编码错误，请使用 UTF-8")

    model, vocab, id2label = _get_model()
    results = []
    for line in lines:
        try:
            raw = predict_text(line, model, vocab, id2label)
            preds = [{"class": k, "confidence": v} for k, v in raw.items()]
            preds.sort(key=lambda x: x["confidence"], reverse=True)
            results.append({"text": line, "results": preds})
            _add_history(line, preds, "批量")
        except Exception as e:
            results.append({"text": line, "results": [], "error": str(e)})
    return {"results": results}


# ─── 历史记录 ────────────────────────────────────

@router.get("/history")
def get_history():
    with _history_lock:
        out = []
        for h in reversed(_history):
            top1 = h["results"][0] if h["results"] else None
            out.append({
                "id": h["id"],
                "text": h["text"][:60] + "..." if len(h["text"]) > 60 else h["text"],
                "timestamp": h["timestamp"],
                "source": h["source"],
                "top1": top1["class"] if top1 else None,
                "top1_conf": top1["confidence"] if top1 else None,
            })
        return {"history": out}


@router.get("/history/{entry_id}")
def get_history_detail(entry_id: int):
    with _history_lock:
        for h in _history:
            if h["id"] == entry_id:
                return {
                    "id": h["id"],
                    "text": h["text"],
                    "timestamp": h["timestamp"],
                    "source": h["source"],
                    "results": h["results"],
                }
    raise HTTPException(404, "记录不存在")


@router.delete("/history")
def clear_history():
    global _history
    with _history_lock:
        _history.clear()
    return {"ok": True}
