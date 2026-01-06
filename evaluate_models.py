"""
evaluate_models.py — HAC-VT standalone evaluation (Step A.3)

Creates ONE master comparison table with:
Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Neutral F1, Time (ms/review)

Inputs:
- test.csv in repo root (or change TEST_CSV path)
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parent
TEST_CSV = REPO_ROOT / "test.csv"   # you already have this file
OUT_CSV = REPO_ROOT / "Table_5_1_Master_Comparison.csv"

LABELS = ["neg", "neu", "pos"]


# -----------------------------
# Helpers
# -----------------------------
def find_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try sensible defaults, then fall back with a clear error.
    """
    text_candidates = ["text", "review", "review_text", "sentence", "content"]
    label_candidates = ["label", "sentiment", "y", "target", "class"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect text/label columns.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Expected text in {text_candidates} and label in {label_candidates}.\n"
            f"Fix: rename your columns or edit find_text_and_label_columns()."
        )

    return text_col, label_col


def avg_time_ms_per_review(predict_fn: Callable[[List[str]], List[str]], texts: List[str],
                           n_runs: int = 3, warmup: int = 1) -> float:
    """
    Average inference time per review in ms.
    """
    # warmup
    for _ in range(warmup):
        _ = predict_fn(texts)

    times = []
    for _ in range(n_runs):
        t0 = perf_counter()
        _ = predict_fn(texts)
        t1 = perf_counter()
        times.append(t1 - t0)

    avg_seconds = float(np.mean(times))
    return (avg_seconds / max(len(texts), 1)) * 1000.0


def build_master_table(y_true: List[str], texts: List[str],
                       model_predict_fns: Dict[str, Callable[[List[str]], List[str]]]) -> pd.DataFrame:
    rows = []

    for name, predict_fn in model_predict_fns.items():
        y_pred = predict_fn(texts)

        acc = accuracy_score(y_true, y_pred)
        p_macro = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        r_macro = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

        # Neutral F1 only
        f1_neu = f1_score(y_true, y_pred, labels=["neu"], average=None, zero_division=0)[0]

        t_ms = avg_time_ms_per_review(predict_fn, texts, n_runs=3, warmup=1)

        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Macro-Precision": p_macro,
            "Macro-Recall": r_macro,
            "Macro-F1": f1_macro,
            "Neutral F1": f1_neu,
            "Time (ms/review)": t_ms,
        })

    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    return df


# -----------------------------
# Model Predictors
# -----------------------------
def vader_predict_batch(texts: List[str]) -> List[str]:
    """
    Uses VADER compound score:
    - compound <= -0.05 -> neg
    - compound >=  0.05 -> pos
    - else -> neu
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = analyzer.polarity_scores(str(t))["compound"]
        if c <= -0.05:
            out.append("neg")
        elif c >= 0.05:
            out.append("pos")
        else:
            out.append("neu")
    return out


def textblob_predict_batch(texts: List[str]) -> List[str]:
    """
    Uses TextBlob polarity:
    - polarity < 0 -> neg
    - polarity > 0 -> pos
    - else -> neu
    """
    from textblob import TextBlob

    out = []
    for t in texts:
        p = TextBlob(str(t)).sentiment.polarity
        if p < 0:
            out.append("neg")
        elif p > 0:
            out.append("pos")
        else:
            out.append("neu")
    return out


def hacvt_predict_batch(texts: List[str]) -> List[str]:
    """
    Calls your HAC-VT model from hacvt.model.
    This function is written to be robust: it tries common method names.
    """
    from hacvt.model import HACVTModel

    model = HACVTModel()  # should load default config/model assets

    preds = []
    for t in texts:
        # Try common prediction methods
        if hasattr(model, "predict_one"):
            y = model.predict_one(str(t))
        elif hasattr(model, "predict"):
            y = model.predict(str(t))
        else:
            raise AttributeError("HACVTModel has no predict_one() or predict() method.")

        # Normalise to expected labels
        y = str(y).strip().lower()
        if y in ["negative", "neg"]:
            preds.append("neg")
        elif y in ["neutral", "neu"]:
            preds.append("neu")
        elif y in ["positive", "pos"]:
            preds.append("pos")
        else:
            # If your model returns numeric class ids, map them here
            raise ValueError(f"Unexpected HAC-VT label output: {y!r}")

    return preds


# -----------------------------
# Main
# -----------------------------
def main():
    print("=== HAC-VT Master Comparison Table — Step A.3 ===")
    print("Repo root:", REPO_ROOT)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing {TEST_CSV}. Put your test split CSV at repo root or update TEST_CSV path.")

    df = pd.read_csv(TEST_CSV)
    text_col, label_col = find_text_and_label_columns(df)

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(str).str.lower().tolist()

    # Basic label sanity
    allowed = set(LABELS)
    bad = sorted(set(y_true) - allowed)
    if bad:
        raise ValueError(
            f"Labels in {label_col} must be {LABELS}. Found unexpected labels: {bad}\n"
            f"Fix: map your labels to neg/neu/pos in the CSV."
        )

    print(f"Loaded {len(df)} rows from {TEST_CSV.name}")
    print(f"Using text column: {text_col}")
    print(f"Using label column: {label_col}")

    model_predict_fns = {
        "VADER": vader_predict_batch,
        "TextBlob": textblob_predict_batch,
        "HAC-VT": hacvt_predict_batch,
    }

    table = build_master_table(y_true=y_true, texts=texts, model_predict_fns=model_predict_fns)

    # Print summary
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\nMaster Comparison Table:")
        print(table)

    table.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
