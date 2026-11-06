"""Feature preparation shared by training and the API, so both see exactly the same inputs."""

import re

import numpy as np
import pandas as pd

# columns that are never model inputs
NOT_FEATURES = {"TransactionID", "isFraud", "TransactionDT", "day"}


def add_features(df):
    """Add a few simple features on top of the raw columns."""
    out = df.copy()
    seconds = out["TransactionDT"].astype("float64")
    out["hour"] = (seconds // 3600) % 24
    out["weekday"] = (seconds // 86400) % 7
    amount = out["TransactionAmt"].astype("float64")
    out["amt_cents"] = (amount - np.floor(amount)).round(3)
    if "P_emaildomain" in out and "R_emaildomain" in out:
        same = out["P_emaildomain"] == out["R_emaildomain"]
        out["same_email_domain"] = same.where(out["R_emaildomain"].notna()).astype("float64")
    return out


def to_model_input(df, features, categories):
    """Turn raw rows into the exact table the model was trained on."""
    data = add_features(df)
    for col in features:
        if col not in data:
            data[col] = np.nan
    data = data[features]
    for col, cats in categories.items():
        data[col] = pd.Categorical(data[col], categories=cats)
    for col in features:
        if col not in categories:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("float32")
    return data


ID_NAMES = {"id_30": "Operating system", "id_31": "Browser", "id_33": "Screen size",
            "id_19": "Identity id_19", "id_20": "Identity id_20", "id_13": "Identity id_13"}


def describe(feature, value):
    """Plain-English label for a feature and its value, used to explain a score."""
    shown = "missing" if value is None or (isinstance(value, float) and np.isnan(value)) else value
    if isinstance(shown, float):
        shown = f"{shown:g}"
    labels = {
        "TransactionAmt": f"Amount: ${shown}",
        "ProductCD": f"Product type: {shown}",
        "card1": f"Card group (card1): {shown}",
        "card2": f"Card info (card2): {shown}",
        "card3": f"Card info (card3): {shown}",
        "card4": f"Card network: {shown}",
        "card5": f"Card info (card5): {shown}",
        "card6": f"Card type: {shown}",
        "addr1": f"Billing region: {shown}",
        "addr2": f"Billing country: {shown}",
        "P_emaildomain": f"Buyer email domain: {shown}",
        "R_emaildomain": f"Recipient email domain: {shown}",
        "DeviceType": f"Device type: {shown}",
        "DeviceInfo": f"Device: {shown}",
        "hour": f"Hour of day: {shown}",
        "weekday": f"Day of week (0-6): {shown}",
        "amt_cents": f"Cents in amount: {shown}",
        "same_email_domain": f"Buyer and recipient same email domain: {shown}",
    }
    if feature in labels:
        return labels[feature]
    if feature in ID_NAMES:
        return f"{ID_NAMES[feature]}: {shown}"
    group = re.match(r"^(C|D|M|V|dist|id_)\d+$", feature)
    kinds = {"C": "Count", "D": "Days since", "M": "Match check", "V": "Vesta feature",
             "dist": "Distance", "id_": "Identity field"}
    if group:
        return f"{kinds[group.group(1)]} {feature}: {shown}"
    return f"{feature}: {shown}"
