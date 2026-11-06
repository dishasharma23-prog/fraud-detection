"""FastAPI service: scores a card transaction and explains the score."""

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fraud.features import describe, to_model_input

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model"))

model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
meta = json.loads((MODEL_DIR / "meta.json").read_text())
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
samples = json.loads((MODEL_DIR / "samples.json").read_text())
samples_by_id = {s["id"]: s for s in samples}
FEATURES, CATEGORIES, THRESHOLD = meta["features"], meta["categories"], meta["threshold"]

app = FastAPI(title="Fraud Detection API", version="1.0")


class Transaction(BaseModel):
    values: dict


def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def score(values):
    """Probability of fraud, the decision at the alert threshold, and the top reasons."""
    if "TransactionAmt" not in values or "TransactionDT" not in values:
        raise HTTPException(422, "values must include TransactionAmt and TransactionDT")
    row = pd.DataFrame([values])
    for col in CATEGORIES:
        if col in row:
            row[col] = row[col].map(lambda v: None if v in (None, "") else str(v))
    X = to_model_input(row, FEATURES, CATEGORIES)

    probability = float(model.predict(X)[0])
    # SHAP values from LightGBM itself: one contribution per feature, plus a baseline at the end
    contributions = model.predict(X, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(contributions))[:6]
    raw = X.iloc[0]
    reasons = [{
        "feature": FEATURES[i],
        "label": describe(FEATURES[i], clean(raw.iloc[i])),
        "impact": round(float(contributions[i]), 4),
    } for i in order if contributions[i] != 0]

    return {
        "fraud_probability": round(probability, 4),
        "flagged": probability >= THRESHOLD,
        "threshold": round(THRESHOLD, 4),
        "reasons": reasons,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.get("/api/samples")
def list_samples():
    return [{
        "id": s["id"],
        "is_fraud": s["is_fraud"],
        "amount": s["values"].get("TransactionAmt"),
        "product": s["values"].get("ProductCD"),
        "card": s["values"].get("card4"),
    } for s in samples]


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: int):
    if sample_id not in samples_by_id:
        raise HTTPException(404, "no such sample")
    return samples_by_id[sample_id]


@app.post("/api/score")
def score_transaction(transaction: Transaction):
    return score(transaction.values)


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "app" / "static" / "index.html")
