# examples/evaluate_dataset.py
"""
Dataset evaluation example (HAC-VT): ENFORCES tau calibration.

Rule enforced here:
- If profile has no calibrated tau -> STOP and show:
  "Please calibrate tau using a labelled dev set before evaluation."

This file is intentionally minimal and model-agnostic.
You will later replace `dummy_decision_fn` with your real HAC-VT decision function.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from hacvt.io import load_profile, require_calibrated_tau
from hacvt.calibration import apply_tau


def load_labelled_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    Expects a CSV with headers including at least:
      - text
      - label   (values: neg, neu, pos)

    Example:
      text,label
      "Great product",pos
      "Not good",neg
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    texts: List[str] = []
    labels: List[str] = []

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row. Expected columns: text,label")

        fieldnames = [c.strip().lower() for c in reader.fieldnames]
        if "text" not in fieldnames or "label" not in fieldnames:
            raise ValueError(f"CSV must contain columns 'text' and 'label'. Found: {reader.fieldnames}")

        # map original keys (case-preserving) to normalized
        # so we can read even if headers are Text/Label etc.
        key_map = {name.strip().lower(): name for name in reader.fieldnames}
        text_key = key_map["text"]
        label_key = key_map["label"]

        for row in reader:
            t = (row.get(text_key) or "").strip()
            y = (row.get(label_key) or "").strip().lower()

            if not t:
                continue  # skip empty text rows
            if y not in {"neg", "neu", "pos"}:
                raise ValueError(f"Invalid label '{y}'. Use only: neg, neu, pos.")

            texts.append(t)
            labels.append(y)

    if not texts:
        raise ValueError("No valid rows found in CSV (empty or missing text).")

    return texts, labels


# -------------------------------------------------------------------
# Replace this with your real HAC-VT decision function later.
# It must return a float decision value (e.g., pos_ll - neg_ll).
# -------------------------------------------------------------------
def dummy_decision_fn(text: str) -> float:
    """
    Placeholder scoring.
    DO NOT use for real results.
    """
    t = text.lower()
    if "good" in t or "great" in t or "excellent" in t:
        return 0.8
    if "bad" in t or "poor" in t or "terrible" in t:
        return -0.8
    return 0.0


def macro_f1_3class(y_true: List[str], y_pred: List[str]) -> float:
    """
    Small macro-F1 helper (no sklearn dependency).
    """
    labels = ["neg", "neu", "pos"]

    def f1_for(lbl: str) -> float:
        tp = sum((yt == lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lbl and yp != lbl) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return sum(f1_for(l) for l in labels) / 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Path to labelled test CSV with columns: text,label")
    ap.add_argument("--profile", required=True, help="Path to profile.json")
    args = ap.parse_args()

    # 1) Load profile
    profile = load_profile(args.profile)

    # 2) ENFORCEMENT: require calibrated tau before evaluation
    #    If tau is missing -> raises ValueError with the user-facing message.
    tau = require_calibrated_tau(profile)

    # 3) Load test set
    x_test, y_test = load_labelled_csv(args.test)

    # 4) Predict using tau + decision_fn
    decisions = [dummy_decision_fn(t) for t in x_test]
    y_pred = [apply_tau(d, tau) for d in decisions]

    # 5) Report
    mf1 = macro_f1_3class(y_test, y_pred)
    print("=== HAC-VT Evaluation (enforced tau) ===")
    print(f"Profile: {profile.get('name', 'unknown')}")
    print(f"Tau: {tau:.4f}")
    print(f"Macro-F1: {mf1:.4f}")
    print(f"N_test: {len(x_test)}")


if __name__ == "__main__":
    main()
