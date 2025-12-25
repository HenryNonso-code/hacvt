# hacvt/calibration.py
"""
HAC-VT calibration: learn neutral-band tau on a development set.

Core idea:
- For each text, HAC-VT produces a scalar decision value "d"
  (e.g., log-likelihood difference: pos_ll - neg_ll).
- With tau >= 0:
    if d >  tau  -> positive
    if d < -tau  -> negative
    else         -> neutral

This module learns tau on a dev set by maximising a metric (default: macro-F1).

This file intentionally does NOT implement the full scorer (NB / lexicon hybrid).
It only needs a way to obtain a decision value per text via a caller-supplied function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import math

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None


Label = str
DecisionFn = Callable[[str], float]


@dataclass(frozen=True)
class TauCalibrationResult:
    tau: float
    metric_name: str
    metric_value: float
    grid: List[Tuple[float, float]]  # list of (tau, metric_value)
    n_dev: int


def apply_tau(decision_value: float, tau: float) -> Label:
    """
    Convert a decision scalar into a 3-class label using tau.
    """
    if decision_value > tau:
        return "pos"
    if decision_value < -tau:
        return "neg"
    return "neu"


def _validate_labels(y: Sequence[Label]) -> None:
    allowed = {"neg", "neu", "pos"}
    bad = sorted({v for v in y if v not in allowed})
    if bad:
        raise ValueError(
            f"Invalid labels in y_dev: {bad}. Expected only {sorted(allowed)}."
        )


def _macro_f1(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """
    Macro-F1 across the three labels in fixed order.
    Uses sklearn if available; otherwise a small fallback.
    """
    labels = ["neg", "neu", "pos"]

    if f1_score is not None:
        return float(f1_score(y_true, y_pred, labels=labels, average="macro"))

    # Fallback macro-F1 (no sklearn)
    def prf_for(lbl: Label) -> float:
        tp = sum((yt == lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lbl and yp != lbl) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return sum(prf_for(l) for l in labels) / 3.0


def _make_grid_from_decisions(
    decisions: Sequence[float],
    n_grid: int = 101,
    max_tau: Optional[float] = None,
) -> List[float]:
    """
    Create a tau grid using the distribution of |decision|.
    Default: grid from 0 to percentile-based upper bound.
    """
    abs_vals = [abs(d) for d in decisions if d is not None and not math.isnan(d)]
    if not abs_vals:
        # Degenerate case: all NaN/None decisions; only tau=0 makes sense
        return [0.0]

    abs_vals_sorted = sorted(abs_vals)

    # Choose an upper bound that is robust to extreme outliers
    if max_tau is None:
        # 95th percentile as a reasonable cap
        idx = int(0.95 * (len(abs_vals_sorted) - 1))
        upper = abs_vals_sorted[idx]
    else:
        upper = float(max_tau)

    upper = max(0.0, upper)

    if n_grid <= 1 or upper == 0.0:
        return [0.0]

    step = upper / (n_grid - 1)
    return [i * step for i in range(n_grid)]


def calibrate_tau(
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    metric: str = "macro_f1",
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    return_grid: bool = True,
) -> TauCalibrationResult:
    """
    Learn tau on a dev set by grid-searching tau values and maximising a metric.

    Parameters
    ----------
    x_dev, y_dev:
        Development texts and gold labels in {"neg","neu","pos"}.
    decision_fn:
        Callable that maps text -> decision scalar (e.g., pos_ll - neg_ll).
    metric:
        Currently supports "macro_f1" (default). Easily extendable.
    n_grid:
        Number of tau points to test (default 101).
    max_tau:
        Optional manual cap for tau. If None, uses ~95th percentile of |decision|.
    return_grid:
        If True, store full (tau, metric) curve for reporting/plotting.

    Returns
    -------
    TauCalibrationResult with best tau and metric.
    """
    if len(x_dev) != len(y_dev):
        raise ValueError("x_dev and y_dev must have the same length.")
    if len(x_dev) == 0:
        raise ValueError("Dev set is empty; cannot calibrate tau.")

    _validate_labels(y_dev)

    # Compute decisions once
    decisions: List[float] = []
    for t in x_dev:
        d = float(decision_fn(t))
        decisions.append(d)

    # Prepare tau grid
    tau_grid = _make_grid_from_decisions(decisions, n_grid=n_grid, max_tau=max_tau)

    if metric.lower() != "macro_f1":
        raise ValueError(f"Unsupported metric: {metric}. Use metric='macro_f1'.")

    best_tau = 0.0
    best_score = -1.0
    curve: List[Tuple[float, float]] = []

    for tau in tau_grid:
        y_pred = [apply_tau(d, tau) for d in decisions]
        score = _macro_f1(y_dev, y_pred)

        if return_grid:
            curve.append((float(tau), float(score)))

        # Tie-breaker: if same score, prefer smaller tau (more decisive, less neutral inflation)
        if (score > best_score) or (math.isclose(score, best_score) and tau < best_tau):
            best_score = float(score)
            best_tau = float(tau)

    return TauCalibrationResult(
        tau=best_tau,
        metric_name="macro_f1",
        metric_value=best_score,
        grid=curve if return_grid else [],
        n_dev=len(x_dev),
    )


def fit_profile_tau(
    profile: Any,
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    n_grid: int = 101,
    max_tau: Optional[float] = None,
) -> Tuple[Any, TauCalibrationResult]:
    """
    Convenience wrapper that learns tau and writes it into the provided profile object.

    Assumes profile has attribute: profile.tau
    Returns (profile, calibration_result).
    """
    result = calibrate_tau(
        x_dev=x_dev,
        y_dev=y_dev,
        decision_fn=decision_fn,
        n_grid=n_grid,
        max_tau=max_tau,
        metric="macro_f1",
        return_grid=True,
    )

    # Write into profile
    try:
        setattr(profile, "tau", result.tau)
    except Exception as e:
        raise TypeError(
            "Profile object does not support setting attribute 'tau'."
        ) from e

    return profile, result
