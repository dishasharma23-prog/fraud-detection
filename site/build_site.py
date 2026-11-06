"""Builds the static demo in docs/: runs the trained model on the demo transactions and saves the results.

    python site/build_site.py            # uses model/ and writes docs/data.json
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_DIR", str(ROOT / "model"))

from app.main import metrics, samples, score  # noqa: E402  (loads the model once)

SHOWN = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]

transactions = []
for sample in samples:
    result = score(sample["values"])
    transactions.append({
        "id": sample["id"],
        "is_fraud": sample["is_fraud"],
        "fields": {k: sample["values"].get(k) for k in SHOWN},
        "result": result,
    })

docs = ROOT / "docs"
docs.mkdir(exist_ok=True)
(docs / "data.json").write_text(json.dumps({"metrics": metrics, "transactions": transactions}))
for name in ["index.html", "site.js"]:
    shutil.copy(ROOT / "site" / name, docs / name)
(docs / ".nojekyll").write_text("")

caught = sum(t["result"]["flagged"] for t in transactions if t["is_fraud"])
print(f"wrote {len(transactions)} transactions to docs/ ({caught} of {sum(t['is_fraud'] for t in transactions)} fraud flagged)")
