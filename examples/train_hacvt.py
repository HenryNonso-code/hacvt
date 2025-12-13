# examples/train_hacvt.py
import json
from pathlib import Path

import pandas as pd

from hacvt import HACVT

# ============================
# Paths
# ============================
# Project root: C:\Users\User\hacvt-package
ROOT = Path(__file__).resolve().parent.parent

# Update this filename if yours is different
DATA_PATH = ROOT / "data" / "car_reviews_dataset.csv"

# Output model file in the ROOT folder
MODEL_PATH = ROOT / "hacvt_model.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Create ROOT/data/ and place your CSV there, e.g.\n"
        r"C:\Users\User\hacvt-package\data\car_reviews_dataset.csv"
    )

# ============================
# Load CSV
# ============================
df = pd.read_csv(DATA_PATH)

# ============================
# Auto-detect text & label columns
# ============================
CANDIDATE_TEXT_COLS = [
    "text", "review", "content", "body",
    "comment", "description", "title"
]

TEXT_COL = None
for c in CANDIDATE_TEXT_COLS:
    if c in df.columns:
        TEXT_COL = c
        break

if TEXT_COL is None:
    raise ValueError(
        f"No text column found. Available columns: {list(df.columns)}\n"
        f"Expected one of: {CANDIDATE_TEXT_COLS}"
    )

LABEL_COL = "rating" if "rating" in df.columns else "label"
if LABEL_COL not in df.columns:
    raise ValueError(
        f"No label/rating column found. Available columns: {list(df.columns)}\n"
        "Expected a 'rating' column (1–5) or a 'label' column (neg/neu/pos)."
    )

# ============================
# Clean missing/invalid rows
# ============================
# Ensure text is string and trimmed
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

# Drop empty texts
df = df[df[TEXT_COL].notna() & (df[TEXT_COL] != "")]

# Drop missing labels
df = df[df[LABEL_COL].notna()]

# If using numeric ratings, coerce and filter to valid range
if LABEL_COL == "rating":
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df[df[LABEL_COL].notna()]
    df = df[(df[LABEL_COL] >= 1) & (df[LABEL_COL] <= 5)]

# Final safety checks
if df.empty:
    raise ValueError(
        "After cleaning, no rows remain for training.\n"
        f"Check your {TEXT_COL} and {LABEL_COL} columns."
    )

# ============================
# Prepare training lists
# ============================
texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].tolist()

print("Using TEXT_COL:", TEXT_COL)
print("Using LABEL_COL:", LABEL_COL)
print("Training rows:", len(df))

# ============================
# Train HAC-VT
# ============================
model = HACVT()
model.fit(texts, labels)

print("Dev macro-F1:", model.dev_macro_f1_)
print("tau_low:", model.tau_low_, "tau_high:", model.tau_high_)

# ============================
# Save trained model
# ============================
with MODEL_PATH.open("w", encoding="utf-8") as f:
    json.dump(model.to_dict(), f)

print(f"Saved trained model to: {MODEL_PATH}")
