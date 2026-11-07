"""API tests: normal scoring, bad input, and a regression check against the published demo results."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import THRESHOLD, app, samples

client = TestClient(app)
DEMO = json.loads((Path(__file__).resolve().parent.parent / "docs" / "data.json").read_text())


def first_sample():
    return dict(samples[0]["values"])


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_are_sane():
    m = client.get("/api/metrics").json()
    assert 0 < m["pr_auc_random_baseline"] < m["pr_auc"] <= 1
    assert 0.5 < m["roc_auc"] <= 1
    assert m["at_threshold"]["false_positive_rate"] < 0.05


def test_score_shape_and_ranges():
    r = client.post("/api/score", json={"values": first_sample()})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["flagged"] == (body["fraud_probability"] >= round(THRESHOLD, 4))
    assert 1 <= len(body["reasons"]) <= 6
    impacts = [abs(x["impact"]) for x in body["reasons"]]
    assert impacts == sorted(impacts, reverse=True)


@pytest.mark.parametrize("missing", ["TransactionAmt", "TransactionDT"])
def test_missing_required_field_is_rejected(missing):
    values = first_sample()
    del values[missing]
    assert client.post("/api/score", json={"values": values}).status_code == 422


@pytest.mark.parametrize("amount", ["abc", -5, None])
def test_bad_amount_is_rejected(amount):
    values = first_sample()
    values["TransactionAmt"] = amount
    assert client.post("/api/score", json={"values": values}).status_code == 422


def test_body_without_values_is_rejected():
    assert client.post("/api/score", json={"amount": 10}).status_code == 422


def test_unknown_category_value_still_scores():
    values = first_sample()
    values.update(ProductCD="NEW_PRODUCT", card4="unknown-network", DeviceInfo="never seen before")
    r = client.post("/api/score", json={"values": values})
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_probability"] <= 1


def test_empty_string_counts_as_missing():
    values = first_sample()
    a = client.post("/api/score", json={"values": {**values, "P_emaildomain": ""}}).json()
    values.pop("P_emaildomain", None)
    b = client.post("/api/score", json={"values": values}).json()
    assert a["fraud_probability"] == b["fraud_probability"]


def test_minimal_transaction_scores():
    r = client.post("/api/score", json={"values": {"TransactionAmt": 49.99, "TransactionDT": 86400}})
    assert r.status_code == 200


def test_unknown_sample_is_404():
    assert client.get("/api/samples/1").status_code == 404


def test_scores_match_published_demo():
    """Regression check: the API must give the same scores as the results published on the demo page."""
    by_id = {s["id"]: s for s in samples}
    for t in DEMO["transactions"][:20]:
        r = client.post("/api/score", json={"values": by_id[t["id"]]["values"]}).json()
        assert r["fraud_probability"] == pytest.approx(t["result"]["fraud_probability"], abs=1e-4)
        assert r["flagged"] == t["result"]["flagged"]
