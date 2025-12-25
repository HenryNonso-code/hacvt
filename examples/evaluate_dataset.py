# examples/evaluate_dataset.py
"""
Dataset evaluation example (HAC-VT): ENFORCES tau calibration.

Rule enforced here:
- If profile has no calibrated tau -> STOP and show:
  "Please calibrate tau using a labelled dev set before evaluation."

This script uses the REAL HACVT model numeric decision value (delta)
and applies the calibrated tau to produce 3-class predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hacvt.io import load_profile, require_calibrated_tau
from hacvt.calibration import apply_tau
from hacvt.model import HACVT


# ----------------------------
# CSV loader (test set)
# ----------------------------
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

        key_map = {name.strip().lower(): name for name in reader.fieldnames}
        text_key = key_map["text"]
        label_key = key_map["label"]

        for row in reader:
            t = (row.get(text_key) or "").strip()
            y = (row.get(label_key) or "").strip().lower()

            if not t:
                continue
            if y not in {"neg", "neu", "pos"}:
                raise ValueError(f"Invalid label '{y}'. Use only: neg, neu, pos.")

            texts.append(t)
            labels.append(y)

    if not texts:
        raise ValueError("No valid rows found in CSV (empty or missing text).")

    return texts, labels


# ----------------------------
# Model loader
# ----------------------------
def _find_model_path(profile: Dict[str, Any], explicit_path: Optional[str] = None) -> Path:
    """
    Resolve model JSON path in priority order:
      1) --model argument (explicit_path)
      2) profile.meta.model_source
      3) repo root: hacvt_model.json
      4) package: hacvt/default_model.json
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Model JSON not found at: {explicit_path}")
        return p

    meta = profile.get("meta", {}) if isinstance(profile.get("meta", {}), dict) else {}
    ms = meta.get("model_source")
    if isinstance(ms, str) and ms.strip():
        p = Path(ms)
        if p.exists():
            return p
        # If stored as a posix path from earlier runs, try to interpret relative to repo root too
        p2 = Path(ms.replace("/", "\\"))
        if p2.exists():
            return p2

    candidates = [
        Path("hacvt_model.json"),
        Path("hacvt") / "default_model.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No model JSON found. Expected one of:\n"
        "  - profile.meta.model_source (existing file)\n"
        "  - hacvt_model.json (repo root)\n"
        "  - hacvt/default_model.json (package)\n"
        "Or pass --model path/to/model.json"
    )


def load_hacvt_model(model_path: Path) -> HACVT:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    return HACVT.from_dict(data)


# ----------------------------
# Metrics + diagnostics
# ----------------------------
def macro_f1_3class(y_true: List[str], y_pred: List[str]) -> float:
    labels = ["neg", "neu", "pos"]

    def f1_for(lbl: str) -> float:
        tp = sum((yt == lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lbl and yp != lbl) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return sum(f1_for(l) for l in labels) / 3.0


def _pct(n: int, total: int) -> float:
    return (100.0 * n / total) if total else 0.0


def _dist_report(name: str, labels: List[str]) -> str:
    total = len(labels)
    c = Counter(labels)
    parts = [f"{k}={c.get(k, 0)} ({_pct(c.get(k, 0), total):.1f}%)" for k in ("neg", "neu", "pos")]
    return f"{name}: " + ", ".join(parts)


def confusion_counts_3x3(y_true: List[str], y_pred: List[str]) -> List[List[int]]:
    idx = {"neg": 0, "neu": 1, "pos": 2}
    m = [[0, 0, 0] for _ in range(3)]
    for yt, yp in zip(y_true, y_pred):
        m[idx[yt]][idx[yp]] += 1
    return m


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Path to labelled test CSV with columns: text,label")
    ap.add_argument("--profile", required=True, help="Path to profile.json")
    ap.add_argument("--model", default=None, help="Optional path to model.json (overrides profile + defaults)")
    args = ap.parse_args()

    # 1) Load profile
    profile = load_profile(args.profile)

    # 2) ENFORCEMENT: require calibrated tau before evaluation
    tau = require_calibrated_tau(profile)

    # 3) Load model (real HACVT) + numeric decision (delta)
    model_path = _find_model_path(profile, args.model)
    model = load_hacvt_model(model_path)

    # 4) Load test set
    x_test, y_test = load_labelled_csv(args.test)

    # 5) Predict using REAL delta + apply_tau
    decisions = [float(model.delta(t)) for t in x_test]
    y_pred = [apply_tau(d, float(tau)) for d in decisions]

    # 6) Report
    mf1 = macro_f1_3class(y_test, y_pred)
    cm = confusion_counts_3x3(y_test, y_pred)

    print("=== HAC-VT Evaluation (enforced tau) ===")
    print(f"Profile: {profile.get('name', 'unknown')}")
    print(f"Model: {profile.get('meta', {}).get('model_source', str(model_path))}")
    print(f"Tau: {float(tau):.4f}")
    print(f"Macro-F1: {mf1:.4f}")
    print(f"N_test: {len(x_test)}")
    print(_dist_report("Gold distribution", y_test))
    print(_dist_report("Pred distribution", y_pred))
    print(f"Neutral rate (pred): {_pct(Counter(y_pred).get('neu', 0), len(y_pred)):.1f}%")
    print("Confusion counts (rows=true, cols=pred) order=[neg, neu, pos]:")
    print(cm)


if __name__ == "__main__":
    main()
