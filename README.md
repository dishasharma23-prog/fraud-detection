# Card Fraud Detection

A LightGBM model that scores card transactions for fraud, served by a FastAPI service with a small web page.
Every score comes with the factors that drove it (SHAP values).

**Live demo:** https://dishasharma23-prog.github.io/fraud-detection/ (the model's scores and SHAP explanations for 100 transactions from the test period)

Trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset:
590,540 real e-commerce transactions from Vesta, 3.5% of them fraud.

## How it was built

1. **Time-based split.** Transactions are sorted by time: the oldest 70% train the model, the next 10% are used to stop
   training and pick the alert threshold, and the newest 20% are kept untouched for the final test. A random split would
   let the model peek at the future and inflate the scores.
2. **Model.** LightGBM (gradient-boosted trees) on all ~430 columns, with text columns such as product, card network,
   email domain and device treated as categories, plus four extra features: hour of day, day of week, the cents in the
   amount, and whether buyer and recipient share an email domain.
3. **Metrics that fit rare fraud.** With 3.5% fraud, a model that always says "legit" is 96.5% accurate, so accuracy is
   not reported. The results use PR-AUC, ROC-AUC, and recall when only 1% of legitimate transactions may be flagged.
4. **Threshold.** A transaction is flagged when its probability passes the threshold that flags 1% of legitimate
   transactions in the validation period.
5. **Explanations.** LightGBM computes SHAP values directly (`pred_contrib=True`), so the API returns the top factors
   for every score without a separate SHAP model.

Results on the test period are in `model/metrics.json` and on the live page.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | service check |
| GET | `/api/metrics` | test-set metrics |
| GET | `/api/samples` | demo transactions from the test period |
| GET | `/api/samples/{id}` | one demo transaction with all its values |
| POST | `/api/score` | `{"values": {...}}` → fraud probability, flag, top reasons |

Interactive docs at `/docs`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

14 API tests run on every push (GitHub Actions): normal scoring, missing or invalid fields (text, negative or empty
amounts are rejected with 422), product or card values the model has never seen, empty strings treated as missing,
unknown IDs, and a regression check that the API still returns the exact scores published on the demo page.

## Run it locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

The model files in `model/` (`model.txt.gz`, `meta.json`, `metrics.json`, `samples.json`) come from
`notebooks/train_and_deploy.ipynb`, which trains on the Kaggle data in Google Colab.

Or with Docker:

```bash
docker build -t fraud-detection .
docker run -p 7860:7860 fraud-detection
```

## Project layout

```
fraud/features.py   feature preparation shared by training and the API
fraud/train.py      time split, LightGBM training, metrics, demo samples
app/main.py         FastAPI service
app/static/         web page served by the API
tests/              API tests (pytest)
site/build_site.py  builds the static demo in docs/ (GitHub Pages) from the trained model
notebooks/          Colab notebook that trains the model
model/              trained model and its test results
Dockerfile          container for the API
```

Built by Disha Sharma.
