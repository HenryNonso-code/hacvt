# examples/calibrate_tau.py
"""
Calibrate tau on a labelled dev set and save a canonical profile.json.

Usage (from repo root):
  python -m examples.calibrate_tau --dev dev.csv --out profile.json --name MyProfile

Dev CSV must have columns: text,label
Labels must be: neg, neu, pos
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

from hacvt.calibration import calibrate_tau
from hacvt.io import save_profile


def load_labelled_csv(path: str) -> Tuple[List[str], List[str]]:
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
        raise ValueError("No valid rows found in dev CSV (empty or missing text).")

    return texts, labels


# -------------------------------------------------------------------
# Replace this later with your REAL HAC-VT decision function.
# For now, it exists so tau calibration + profile saving is end-to-end.
# Must return: float decision value (pos_ll - neg_ll style).
# -------------------------------------------------------------------
def dummy_decision_fn(text: str) -> float:
    t = text.lower()
    if "good" in t or "great" in t or "excellent" in t:
        return 0.8
    if "bad" in t or "poor" in t or "terrible" in t:
        return -0.8
    return 0.0


def build_profile(
    name: str,
    tau: float,
    dev_macro_f1: float,
    class_counts: Dict[str, int],
    *,
    dataset: str = "unknown",
) -> Dict[str, Any]:
    """
    Builds a canonical profile object matching the README schema.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "name": name,
        "version": "1.0",
        "tau": float(tau),
        "label_map": {"neg": 0, "neu": 1, "pos": 2},
        "params": {
            "tokenizer": {
                "lowercase": True,
                "keep_punct": False,
                "negation_scope": 3
            },
            "scoring": {
                "decision_rule": "pos_ll_minus_neg_ll"
            }
        },
        "meta": {
            "dataset": dataset,
            "timestamp_utc": ts,
            "text_col": "text",
            "label_col": "label",
            "n_dev": int(sum(class_counts.values()))
        },
        "calibration_report": {
            "metric": "macro_f1",
            "dev_macro_f1": float(dev_macro_f1),
            "tau_grid_size": 101,
            "class_counts_dev": dict(class_counts)
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="Labelled dev CSV (text,label)")
    ap.add_argument("--out", required=True, help="Output profile.json path")
    ap.add_argument("--name", required=True, help="Profile name to store in JSON")
    ap.add_argument("--dataset", default="unknown", help="Dataset name for meta (optional)")
    args = ap.parse_args()

    # 1) Load dev set
    x_dev, y_dev = load_labelled_csv(args.dev)
    counts = Counter(y_dev)

    # 2) Calibrate tau
    result = calibrate_tau(
        x_dev=x_dev,
        y_dev=y_dev,
        decision_fn=dummy_decision_fn,
        metric="macro_f1",
        n_grid=101,
        max_tau=None,
        return_grid=True
    )

    # 3) Build canonical profile + save
    profile = build_profile(
        name=args.name,
        tau=result.tau,
        dev_macro_f1=result.metric_value,
        class_counts={"neg": counts.get("neg", 0), "neu": counts.get("neu", 0), "pos": counts.get("pos", 0)},
        dataset=args.dataset
    )

    save_path = save_profile(profile, args.out)

    print("=== Calibrated profile saved ===")
    print(f"Path: {save_path}")
    print(f"Name: {profile['name']}")
    print(f"Tau: {profile['tau']:.4f}")
    print(f"Dev Macro-F1: {profile['calibration_report']['dev_macro_f1']:.4f}")
    print(f"Dev counts: {profile['calibration_report']['class_counts_dev']}")


if __name__ == "__main__":
    main()
