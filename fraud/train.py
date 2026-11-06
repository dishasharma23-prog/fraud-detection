"""Train the fraud model with a time-based split and save everything the API needs.

    python -m fraud.train path/to/train.parquet model/
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from fraud.features import NOT_FEATURES, add_features, to_model_input

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 128,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_to_onehot": 8,
    "cat_smooth": 20,
    "metric": "average_precision",
    "verbose": -1,
    "seed": 42,
}


def time_split(df, valid_from=0.7, test_from=0.8):
    """Oldest 70% to train, next 10% to tune, newest 20% to test, like a model used on future data."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    a, b = int(len(df) * valid_from), int(len(df) * test_from)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def recall_at_fpr(y, scores, threshold):
    flagged = scores >= threshold
    fraud, legit = y == 1, y == 0
    return {
        "recall": float(flagged[fraud].mean()),
        "false_positive_rate": float(flagged[legit].mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "flagged": int(flagged.sum()),
    }


def main(data_path, out_dir, rounds=1500):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    df = pd.read_parquet(data_path)
    raw_columns = [c for c in df.columns if c not in {"isFraud", "day"}]

    # text columns (product, card network, email domain, device...) become categories
    categories = {}
    for col in raw_columns:
        if col not in NOT_FEATURES and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("object").where(df[col].isna(), df[col].astype(str))
            categories[col] = sorted(df[col].dropna().unique().tolist())

    features = [c for c in add_features(df.head(5)).columns if c not in NOT_FEATURES]
    train, valid, test = time_split(df)
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}  features {len(features)}")

    X_train = to_model_input(train, features, categories)
    X_valid = to_model_input(valid, features, categories)
    X_test = to_model_input(test, features, categories)
    y_train, y_valid, y_test = (p["isFraud"].to_numpy() for p in (train, valid, test))

    model = lgb.train(
        PARAMS,
        lgb.Dataset(X_train, y_train),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    p_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    p_test = model.predict(X_test, num_iteration=model.best_iteration)

    # pick the alert threshold on the validation month, so the test month stays untouched
    fpr, tpr, thresholds = roc_curve(y_valid, p_valid)
    threshold = float(thresholds[np.where(fpr <= 0.01)[0][-1]])

    test_fpr, test_tpr, _ = roc_curve(y_test, p_test)
    metrics = {
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc_random_baseline": float(y_test.mean()),
        "recall_at_1pct_fpr": float(test_tpr[np.where(test_fpr <= 0.01)[0][-1]]),
        "at_threshold": recall_at_fpr(y_test, p_test, threshold),
        "threshold": threshold,
        "best_iteration": int(model.best_iteration),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_fraud_cases": int(y_test.sum()),
        "features": len(features),
        "training_minutes": round((time.time() - start) / 60, 1),
    }
    print(json.dumps(metrics, indent=2))

    model.save_model(out / "model.txt", num_iteration=model.best_iteration)
    meta = {"features": features, "categories": categories, "threshold": threshold, "raw_columns": raw_columns}
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # demo transactions from the test period: a mix of caught, missed and normal ones
    test = test.assign(score=p_test)
    fraud = test[test.isFraud == 1]
    legit = test[test.isFraud == 0]
    picks = pd.concat([
        fraud[fraud.score >= threshold].sample(min(30, (fraud.score >= threshold).sum()), random_state=1),
        fraud[fraud.score < threshold].sample(min(15, (fraud.score < threshold).sum()), random_state=1),
        legit[legit.score >= threshold].sample(min(10, (legit.score >= threshold).sum()), random_state=1),
        legit[legit.score < threshold].sample(min(45, (legit.score < threshold).sum()), random_state=1),
    ]).sample(frac=1, random_state=2)
    samples = []
    for _, row in picks.iterrows():
        values = {c: (v.item() if hasattr(v, "item") else v) for c, v in row[raw_columns].items() if pd.notna(v)}
        samples.append({"id": int(row["TransactionID"]), "is_fraud": int(row["isFraud"]), "values": values})
    (out / "samples.json").write_text(json.dumps(samples))
    print(f"saved model, meta, metrics and {len(samples)} samples to {out}")
    return metrics


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
