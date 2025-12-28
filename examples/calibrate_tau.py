# examples/calibrate_tau.py
"""
Calibrate tau on a labelled dev set and save a canonical profile.json.

Usage (from repo root):
  python -m examples.calibrate_tau --dev dev.csv --out profile.json --name MyProfile --dataset CarReviews

Dev CSV must have columns: text,label
Labels must be: neg, neu, pos
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

from hacvt.calibration import calibrate_tau
from hacvt.io import save_profile
from hacvt.model import HACVT


# ----------------------------
# CSV loader (dev set)
# ----------------------------
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


# ----------------------------
# Load HACVT model once
# ----------------------------
def _find_model_path(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Model JSON not found at: {explicit_path}")
        return p

    candidates = [
        Path("hacvt_model.json"),
        Path("hacvt") / "default_model.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No model JSON found. Expected one of:\n"
        "  - hacvt_model.json (repo root)\n"
        "  - hacvt/default_model.json (package)\n"
        "Or pass --model path/to/model.json"
    )


def load_hacvt_model(model_path: Path) -> HACVT:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    return HACVT.from_dict(data)


# ----------------------------
# Decision function (numeric) + centering
# ----------------------------
DecisionFn = Callable[[str], float]


def make_raw_decision_fn(model: HACVT) -> DecisionFn:
    """
    Returns callable(text)->float decision value.

    Preferred:
      - decision_value(text) -> float
      - delta(text) -> float  (pos_ll - neg_ll)
    """
    dv = getattr(model, "decision_value", None)
    if callable(dv):
        return lambda text: float(dv(text))

    dl = getattr(model, "delta", None)
    if callable(dl):
        return lambda text: float(dl(text))

    raise AttributeError("HACVT model has no decision_value() or delta() method to produce numeric scores.")


def compute_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def make_centered_decision_fn(raw_fn: DecisionFn, delta_mean: float) -> DecisionFn:
    """
    Center the decision values so that "neutral" tends to sit near 0.

    delta_centered = delta_raw - mean(delta_raw on dev set)
    """
    return lambda text: float(raw_fn(text) - delta_mean)


# ----------------------------
# Profile builder (canonical)
# ----------------------------
def build_profile(
    name: str,
    tau: float,
    dev_macro_f1: float,
    class_counts: Dict[str, int],
    *,
    dataset: str = "unknown",
    model_source: str = "unknown",
    delta_mean: float = 0.0,
    calibration_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    policy = dict(calibration_policy) if calibration_policy else {}

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
                "decision_rule": "pos_ll_minus_neg_ll",
                "centering": {
                    "enabled": True,
                    "method": "dev_mean_subtraction"
                }
            }
        },
        "meta": {
            "dataset": dataset,
            "timestamp_utc": ts,
            "text_col": "text",
            "label_col": "label",
            "n_dev": int(sum(class_counts.values())),
            "model_source": model_source
        },
        "calibration_report": {
            "metric": "macro_f1",
            "dev_macro_f1": float(dev_macro_f1),
            "tau_grid_size": 101,
            "class_counts_dev": dict(class_counts),
            "delta_mean": float(delta_mean),
            "policy": policy
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="Labelled dev CSV (text,label)")
    ap.add_argument("--out", required=True, help="Output profile.json path")
    ap.add_argument("--name", required=True, help="Profile name to store in JSON")
    ap.add_argument("--dataset", default="unknown", help="Dataset name for meta (optional)")
    ap.add_argument("--model", default=None, help="Optional path to model.json (overrides defaults)")
    args = ap.parse_args()

    # 1) Load dev set
    x_dev, y_dev = load_labelled_csv(args.dev)
    counts = Counter(y_dev)

    # 2) Load model + raw numeric decision function
    model_path = _find_model_path(args.model)
    model = load_hacvt_model(model_path)
    raw_decision_fn = make_raw_decision_fn(model)

    # 3) Compute raw decisions ON DEV, then mean-center them
    raw_decisions: List[float] = [float(raw_decision_fn(t)) for t in x_dev]
    delta_mean = compute_mean(raw_decisions)
    centered_decision_fn = make_centered_decision_fn(raw_decision_fn, delta_mean)

    # 4) Calibrate tau using CENTERED decisions
    result = calibrate_tau(
        x_dev=x_dev,
        y_dev=y_dev,
        decision_fn=centered_decision_fn,
        metric="macro_f1",
        n_grid=101,
        max_tau=None,
        min_tau=None,
        return_grid=True
    )

    # 5) Pull optional range fields safely (depends on your current hacvt/calibration.py)
    policy: Dict[str, Any] = {}
    for attr in ("min_tau_used", "max_tau_used"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, (int, float)):
                policy[attr] = float(val)

    # If you later add these fields to TauCalibrationResult, we will store + print them too
    for attr in ("max_neutral_rate_used", "neutral_penalty_used", "neutral_rate_at_best"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, (int, float)):
                policy[attr] = float(val)

    # 6) Build canonical profile + save (includes delta_mean)
    profile = build_profile(
        name=args.name,
        tau=float(result.tau),
        dev_macro_f1=float(result.metric_value),
        class_counts={"neg": counts.get("neg", 0), "neu": counts.get("neu", 0), "pos": counts.get("pos", 0)},
        dataset=args.dataset,
        model_source=str(model_path.as_posix()),
        delta_mean=delta_mean,
        calibration_policy=policy,
    )

    save_path = save_profile(profile, args.out)

    # 7) Log
    print("=== Calibrated profile saved ===")
    print(f"Path: {save_path}")
    print(f"Name: {profile['name']}")
    print(f"Model: {profile['meta'].get('model_source')}")
    print(f"Delta mean (dev): {profile['calibration_report']['delta_mean']:.4f}")
    print(f"Tau: {profile['tau']:.4f}")
    print(f"Dev Macro-F1: {profile['calibration_report']['dev_macro_f1']:.4f}")
    print(f"Dev counts: {profile['calibration_report']['class_counts_dev']}")

    if "min_tau_used" in policy and "max_tau_used" in policy:
        print(f"Tau search range: [{policy['min_tau_used']:.2f}, {policy['max_tau_used']:.2f}]")
    elif "max_tau_used" in policy:
        print(f"Tau search max cap: {policy['max_tau_used']:.2f}")

    if "max_neutral_rate_used" in policy:
        print(f"Neutral cap used: {policy['max_neutral_rate_used']:.2f}")
    if "neutral_penalty_used" in policy:
        print(f"Neutral penalty used: {policy['neutral_penalty_used']:.2f}")
    if "neutral_rate_at_best" in policy:
        print(f"Neutral rate at best tau (dev): {policy['neutral_rate_at_best']:.3f}")


if __name__ == "__main__":
    main()
