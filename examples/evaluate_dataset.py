# examples/evaluate_dataset.py
"""
Dataset evaluation example (HAC-VT): ENFORCES tau calibration.

Adds confidence gating:
- First apply delta (optionally centered by profile delta_mean)
- Apply tau band
- If inside neutral band -> only flip away from neutral if margin >= kappa

This reduces neutral ONLY when the model is confident, without forcing.

Step B (Decision-Utility Metrics):
- Polar Accuracy (non-neutral gold only)
- Neutral Precision
- Neutral Recall (optional)
- Coverage (non-neutral prediction rate)

Macro-F1 remains a diagnostic reference line, not the primary objective.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hacvt.io import load_profile, require_calibrated_tau
from hacvt.model import HACVT

LABEL_NEG = "neg"
LABEL_NEU = "neu"
LABEL_POS = "pos"
VALID_LABELS = {LABEL_NEG, LABEL_NEU, LABEL_POS}


# -----------------------------
# IO: labelled CSV (text,label)
# -----------------------------
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
            if y not in VALID_LABELS:
                raise ValueError(f"Invalid label '{y}'. Use only: neg, neu, pos.")

            texts.append(t)
            labels.append(y)

    if not texts:
        raise ValueError("No valid rows found in CSV (empty or missing text).")

    return texts, labels


# -----------------------------
# Model loading helpers
# -----------------------------
def _find_model_path(profile: Dict[str, Any], explicit_path: Optional[str] = None) -> Path:
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
        p2 = Path(ms.replace("/", "\\"))
        if p2.exists():
            return p2

    candidates = [Path("hacvt_model.json"), Path("hacvt") / "default_model.json"]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No model JSON found. Expected one of:\n"
        "  - profile.meta.model_source\n"
        "  - hacvt_model.json (repo root)\n"
        "  - hacvt/default_model.json (package)\n"
        "Or pass --model path/to/model.json"
    )


def load_hacvt_model(model_path: Path) -> HACVT:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    return HACVT.from_dict(data)


def _safe_get_delta_mean(profile: Dict[str, Any]) -> Tuple[float, bool]:
    calib = profile.get("calibration_report", {})
    if isinstance(calib, dict):
        dm = calib.get("delta_mean", None)
        if isinstance(dm, (int, float)):
            return float(dm), True
    return 0.0, False


# -----------------------------
# Classic metrics kept (macro-F1 diagnostic)
# -----------------------------
def macro_f1_3class(y_true: List[str], y_pred: List[str]) -> float:
    labels = [LABEL_NEG, LABEL_NEU, LABEL_POS]

    def f1_for(lbl: str) -> float:
        tp = sum((yt == lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lbl and yp != lbl) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return sum(f1_for(l) for l in labels) / 3.0


def confusion_counts_3x3(y_true: List[str], y_pred: List[str]) -> List[List[int]]:
    idx = {LABEL_NEG: 0, LABEL_NEU: 1, LABEL_POS: 2}
    m = [[0, 0, 0] for _ in range(3)]
    for yt, yp in zip(y_true, y_pred):
        m[idx[yt]][idx[yp]] += 1
    return m


def _pct(n: int, total: int) -> float:
    return (100.0 * n / total) if total else 0.0


def _dist_report(name: str, labels: List[str]) -> str:
    total = len(labels)
    c = Counter(labels)
    parts = [f"{k}={c.get(k, 0)} ({_pct(c.get(k, 0), total):.1f}%)" for k in (LABEL_NEG, LABEL_NEU, LABEL_POS)]
    return f"{name}: " + ", ".join(parts)


# -----------------------------
# Step B: Decision-Utility Metrics
# -----------------------------
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass
class DecisionUtilityReport:
    polar_accuracy_non_neutral_gold: float
    neutral_precision: float
    neutral_recall: float
    coverage_non_neutral_pred: float

    n_total: int
    n_gold_neu: int
    n_gold_polar: int
    n_pred_neu: int
    n_pred_polar: int


def compute_decision_utility_metrics(
    y_true: List[str],
    y_pred: List[str],
    *,
    neutral_label: str = LABEL_NEU,
) -> DecisionUtilityReport:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} vs {len(y_pred)}")

    # Validate labels (fail fast; avoids silent metric corruption)
    bad_true = {y for y in set(y_true) if y not in VALID_LABELS}
    bad_pred = {y for y in set(y_pred) if y not in VALID_LABELS}
    if bad_true:
        raise ValueError(f"Unknown labels in y_true: {bad_true}")
    if bad_pred:
        raise ValueError(f"Unknown labels in y_pred: {bad_pred}")

    n_total = len(y_true)

    gold_neu_mask = [yt == neutral_label for yt in y_true]
    gold_polar_mask = [not b for b in gold_neu_mask]

    pred_neu_mask = [yp == neutral_label for yp in y_pred]
    pred_polar_mask = [not b for b in pred_neu_mask]

    n_gold_neu = sum(gold_neu_mask)
    n_gold_polar = sum(gold_polar_mask)
    n_pred_neu = sum(pred_neu_mask)
    n_pred_polar = sum(pred_polar_mask)

    # (1) Polar Accuracy on gold-polar subset
    if n_gold_polar > 0:
        correct = 0
        total = 0
        for yt, yp, is_polar in zip(y_true, y_pred, gold_polar_mask):
            if not is_polar:
                continue
            total += 1
            if yp == yt:
                correct += 1
        polar_acc = correct / total if total else 0.0
    else:
        polar_acc = 0.0

    # (2) Neutral Precision: of predicted neutral, how many truly neutral
    tp_neu = sum(1 for yt, yp in zip(y_true, y_pred) if (yp == neutral_label and yt == neutral_label))
    neutral_precision = _safe_div(tp_neu, n_pred_neu)

    # (3) Neutral Recall: of true neutral, how many predicted neutral
    neutral_recall = _safe_div(tp_neu, n_gold_neu)

    # (4) Coverage: fraction predicted polar (neg/pos)
    coverage = _safe_div(n_pred_polar, n_total)

    return DecisionUtilityReport(
        polar_accuracy_non_neutral_gold=float(polar_acc),
        neutral_precision=float(neutral_precision),
        neutral_recall=float(neutral_recall),
        coverage_non_neutral_pred=float(coverage),
        n_total=int(n_total),
        n_gold_neu=int(n_gold_neu),
        n_gold_polar=int(n_gold_polar),
        n_pred_neu=int(n_pred_neu),
        n_pred_polar=int(n_pred_polar),
    )


def print_decision_utility_report(r: DecisionUtilityReport) -> None:
    print("\n================ Decision-Utility Metrics (Primary) ================")
    print(f"Polar Accuracy (gold ∈ {{neg,pos}}): {r.polar_accuracy_non_neutral_gold:.4f}")
    print(f"Neutral Precision (pred=neu → gold=neu): {r.neutral_precision:.4f}")
    print(f"Neutral Recall (gold=neu → pred=neu) [optional]: {r.neutral_recall:.4f}")
    print(f"Coverage (pred ∈ {{neg,pos}}): {r.coverage_non_neutral_pred:.4f}")

    print("\n---------------------- Counts (for interpretation) -----------------")
    print(f"N total: {r.n_total}")
    print(f"N gold neutral: {r.n_gold_neu} | N gold polar: {r.n_gold_polar}")
    print(f"N pred neutral: {r.n_pred_neu} | N pred polar: {r.n_pred_polar}")
    print("===================================================================\n")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Labelled test CSV: text,label")
    ap.add_argument("--profile", required=True, help="profile.json")
    ap.add_argument("--model", default=None, help="Optional model.json path")
    ap.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Confidence margin threshold for gating. 0 disables gating. Try 0.5 to 3.0.",
    )
    args = ap.parse_args()

    # 1) Load profile and enforce tau exists
    profile = load_profile(args.profile)
    tau = float(require_calibrated_tau(profile))

    # 2) Optional centering
    delta_mean, used_delta_mean = _safe_get_delta_mean(profile)

    # 3) Load model
    model_path = _find_model_path(profile, args.model)
    model = load_hacvt_model(model_path)

    # 4) Load test
    x_test, y_test = load_labelled_csv(args.test)

    # 5) Predict using gated rule
    y_pred = [
        model.predict_one_gated(
            t,
            tau=tau,
            delta_mean=(delta_mean if used_delta_mean else 0.0),
            kappa=float(args.kappa),
        )
        for t in x_test
    ]

    # 6) Reports: utility-first + macro-F1 diagnostic + confusion matrix
    util = compute_decision_utility_metrics(y_test, y_pred)
    mf1 = macro_f1_3class(y_test, y_pred)
    cm = confusion_counts_3x3(y_test, y_pred)

    print("=== HAC-VT Evaluation (enforced tau + confidence gating) ===")
    print(f"Profile: {profile.get('name', 'unknown')}")
    print(f"Model: {profile.get('meta', {}).get('model_source', str(model_path))}")
    print(f"Tau: {tau:.4f}")
    print(f"Kappa (margin gate): {float(args.kappa):.4f}")

    if used_delta_mean:
        print(f"Delta centering: ON (delta_mean={delta_mean:.4f})")
    else:
        print("Delta centering: OFF (delta_mean missing; using raw delta)")

    print(f"N_test: {len(x_test)}")
    print(_dist_report("Gold distribution", y_test))
    print(_dist_report("Pred distribution", y_pred))
    print(f"Neutral rate (pred): {_pct(Counter(y_pred).get('neu', 0), len(y_pred)):.1f}%")

    # Primary objective: decision-utility metrics
    print_decision_utility_report(util)

    # Diagnostic only: macro-F1 + confusion counts
    print("---------------------- Macro-F1 (diagnostic only) ------------------")
    print(f"Macro-F1: {mf1:.4f}")
    print("Confusion counts (rows=true, cols=pred) order=[neg, neu, pos]:")
    print(cm)


if __name__ == "__main__":
    main()
