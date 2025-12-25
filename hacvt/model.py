"""
hacvt.model — Lightweight HAC-VT sentiment model

Core usage:

    from hacvt import HACVT

    # Option A: Load a packaged default model (plug-and-play)
    model = HACVT.load_default()
    print(model.analyze("The car is good, not terrible."))

    # Option B: Train on your own data (labels = ratings 1–5 or 'neg'/'neu'/'pos')
    model = HACVT()
    model.fit(texts, labels)
    print(model.analyze("Not bad at all"))

    # Save / Load learned model
    json_data = model.to_dict()
    model2 = HACVT.from_dict(json_data)
"""

import json
import math
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Set

# importlib.resources is available in Python 3.8+, but "files" is more reliable with packaged data.
from importlib import resources


# ============================================================
# Tokenisation & Negation Handling
# ============================================================

WORD_RE = re.compile(r"[A-Za-z']+")

NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "hardly", "scarcely", "cannot", "can't",
    "isn't", "dont", "don't", "doesnt", "doesn't", "won't", "wont",
    "wouldn't", "shouldn't", "couldn't", "didn't", "aint", "ain't",
    "neither", "nor"
}

# Note: We do not keep punctuation tokens in WORD_RE, so we cannot "see" '.' ',' etc in the token stream.
# Instead we reset negation at the end of each sentence-like segment by splitting on punctuation first.
_SENT_SPLIT_RE = re.compile(r"[.!?;]+")


def haac_tokenize(text: str) -> List[str]:
    """
    Negation-aware tokeniser used by HAC-VT.

    Example:
        "not good at all" -> ["NOT_good", "at", "all"]
        "I am not happy. great car" -> negation does not leak past the sentence break
    """
    if not text:
        return []

    output: List[str] = []

    # Split into segments to prevent negation leaking across sentence-like boundaries.
    segments = [seg.strip() for seg in _SENT_SPLIT_RE.split(text.lower()) if seg.strip()]

    for seg in segments:
        tokens = WORD_RE.findall(seg)
        negate = False

        for tok in tokens:
            if tok in NEGATION_WORDS:
                negate = True
                continue

            output.append(f"NOT_{tok}" if negate else tok)

            # Attach negation to the *next* sentiment-bearing token only (as per your design).
            # After one attachment, stop negating.
            if negate:
                negate = False

    return output


# ============================================================
# Likelihoods and Δ-Score
# ============================================================

def compute_counts(
    texts: List[str],
    labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    counts: Dict[str, Counter] = {c: Counter() for c in classes}
    totals: Dict[str, int] = {c: 0 for c in classes}

    for text, label in zip(texts, labels):
        toks = haac_tokenize(text)
        counts[label].update(toks)
        totals[label] += len(toks)

    return counts, totals


def compute_log_likelihoods(
    counts: Dict[str, Counter],
    totals: Dict[str, int],
    alpha: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    vocab: Set[str] = set()
    for c in counts:
        vocab.update(counts[c].keys())

    V = len(vocab)
    ll: Dict[str, Dict[str, float]] = {c: {} for c in counts}

    for c in counts:
        total_c = totals[c] + alpha * V
        for tok in vocab:
            ll[c][tok] = math.log((counts[c][tok] + alpha) / total_c)

    return ll, vocab


def delta_for_tokens(tokens: List[str], log_likelihoods: Dict[str, Dict[str, float]]) -> float:
    ll_pos = sum(log_likelihoods["pos"].get(t, 0.0) for t in tokens)
    ll_neg = sum(log_likelihoods["neg"].get(t, 0.0) for t in tokens)
    return ll_pos - ll_neg


def classify_delta(delta: float, tau_low: float, tau_high: float) -> str:
    if delta < tau_low:
        return "neg"
    if delta > tau_high:
        return "pos"
    return "neu"


# ============================================================
# Evaluation Helpers
# ============================================================

def macro_f1(
    true_labels: List[str],
    pred_labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
) -> float:
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for y, yp in zip(true_labels, pred_labels):
        if y == yp:
            per_class[y]["tp"] += 1
        else:
            per_class[y]["fn"] += 1
            per_class[yp]["fp"] += 1

    f1_scores: List[float] = []
    for c in classes:
        tp = per_class[c]["tp"]
        fp = per_class[c]["fp"]
        fn = per_class[c]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def train_dev_split(
    texts: List[str],
    labels: List[str],
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    indices = list(range(len(texts)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split = int(len(indices) * (1.0 - dev_ratio))
    train_idx = indices[:split]
    dev_idx = indices[split:]

    def subset(idxs: List[int], arr: List[Any]) -> List[Any]:
        return [arr[i] for i in idxs]

    return (
        subset(train_idx, texts),
        subset(train_idx, labels),
        subset(dev_idx, texts),
        subset(dev_idx, labels),
    )


def tune_tau(
    deltas: List[float],
    labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
    max_abs: float = 10.0,
    step: float = 0.1,
) -> Tuple[float, float, float]:
    best_f1 = -1.0
    best_t = 0.0

    steps = int(max_abs / step)
    for i in range(steps + 1):
        t = i * step
        tau_low, tau_high = -t, t
        preds = [classify_delta(d, tau_low, tau_high) for d in deltas]
        f1 = macro_f1(labels, preds, classes)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return -best_t, best_t, best_f1


# ============================================================
# HAC-VT Model Class
# ============================================================

class HACVT:
    """
    HAC-VT sentiment model.

    Labels can be:
        * numbers 1–5  (1–2=neg, 3=neu, 4–5=pos)
        * strings 'neg', 'neu', 'pos'

    Explicit usage modes:
        1) Default packaged model:
            model = HACVT.load_default()

        2) Learned model (dataset-adaptive):
            model = HACVT().fit(texts, labels)

        3) Loaded learned model:
            model = HACVT.from_dict(json_data)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_tau: float = 10.0,
        tau_step: float = 0.1,
        dev_ratio: float = 0.2,
        seed: int = 42,
    ):
        # Hyperparameters
        self.alpha = alpha
        self.max_tau = max_tau
        self.tau_step = tau_step
        self.dev_ratio = dev_ratio
        self.seed = seed

        # Model state
        self.classes: Tuple[str, str, str] = ("neg", "neu", "pos")
        self.log_likelihoods_: Optional[Dict[str, Dict[str, float]]] = None
        self.vocab_: Optional[Set[str]] = None

        # Calibration
        self.tau_low_: float = 0.0
        self.tau_high_: float = 0.0
        self.dev_macro_f1_: Optional[float] = None

    # -------------------------------------------------
    # Default loader (packaged model)
    # -------------------------------------------------
    @classmethod
    def load_default(cls) -> "HACVT":
        """
        Load the packaged default HAC-VT model (plug-and-play).

        This never overrides your learned model. It is only used when explicitly called.
        """
        try:
            default_path = resources.files("hacvt").joinpath("default_model.json")
            data = json.loads(default_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(
                "Unable to load default_model.json from the hacvt package. "
                "Confirm it exists at hacvt/default_model.json and is included in the wheel "
                "via pyproject.toml package-data."
            ) from e

        return cls.from_dict(data)

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    @staticmethod
    def _map_label(y: Any) -> str:
        if isinstance(y, str):
            ys = y.strip().lower()
            if ys in {"neg", "negative"}:
                return "neg"
            if ys in {"neu", "neutral"}:
                return "neu"
            if ys in {"pos", "positive"}:
                return "pos"
            raise ValueError(f"Unknown label string: {y}")

        if isinstance(y, (int, float)):
            if y <= 2:
                return "neg"
            if int(round(y)) == 3:
                return "neu"
            return "pos"

        raise ValueError(f"Unsupported label type: {type(y)}")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def fit(self, texts: List[str], labels: List[Any]) -> "HACVT":
        mapped_labels = [self._map_label(y) for y in labels]

        tr_x, tr_y, dev_x, dev_y = train_dev_split(
            texts, mapped_labels, dev_ratio=self.dev_ratio, seed=self.seed
        )

        counts, totals = compute_counts(tr_x, tr_y, self.classes)
        ll, vocab = compute_log_likelihoods(counts, totals, alpha=self.alpha)
        self.log_likelihoods_ = ll
        self.vocab_ = vocab

        dev_deltas = [delta_for_tokens(haac_tokenize(t), self.log_likelihoods_) for t in dev_x]
        tau_low, tau_high, best_f1 = tune_tau(
            dev_deltas, dev_y, classes=self.classes, max_abs=self.max_tau, step=self.tau_step
        )

        self.tau_low_ = tau_low
        self.tau_high_ = tau_high
        self.dev_macro_f1_ = best_f1
        return self

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    def delta(self, text: str) -> float:
        if self.log_likelihoods_ is None:
            raise RuntimeError("HAC-VT model is not fitted/loaded yet. Use fit(), from_dict(), or load_default().")
        toks = haac_tokenize(text)
        return delta_for_tokens(toks, self.log_likelihoods_)

    def predict_one(self, text: str) -> str:
        d = self.delta(text)
        return classify_delta(d, self.tau_low_, self.tau_high_)

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]

    def score(self, texts: List[str], labels: List[Any]) -> float:
        mapped = [self._map_label(y) for y in labels]
        preds = self.predict(texts)
        return macro_f1(mapped, preds, self.classes)

    def analyze(self, text: str) -> Dict[str, Any]:
        d = self.delta(text)
        label = classify_delta(d, self.tau_low_, self.tau_high_)
        return {
            "text": text,
            "delta": d,
            "label": label,
            "tau_low": self.tau_low_,
            "tau_high": self.tau_high_,
        }

    def decision_value(self, text: str) -> float:
        """
        Continuous HAC-VT decision value for tau calibration.

        Definition:
            decision_value = pos_ll - neg_ll  (same as delta)
        """
        return float(self.delta(text))

    # -------------------------------------------------
    # Serialisation
    # -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        if self.log_likelihoods_ is None or self.vocab_ is None:
            raise RuntimeError("HAC-VT model is not fitted/loaded yet. Cannot serialise.")

        return {
            "alpha": self.alpha,
            "max_tau": self.max_tau,
            "tau_step": self.tau_step,
            "dev_ratio": self.dev_ratio,
            "seed": self.seed,
            "classes": list(self.classes),
            "vocab": sorted(list(self.vocab_)),
            "log_likelihoods": self.log_likelihoods_,
            "tau_low": self.tau_low_,
            "tau_high": self.tau_high_,
            "dev_macro_f1": self.dev_macro_f1_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HACVT":
        obj = cls(
            alpha=data.get("alpha", 1.0),
            max_tau=data.get("max_tau", 10.0),
            tau_step=data.get("tau_step", 0.1),
            dev_ratio=data.get("dev_ratio", 0.2),
            seed=data.get("seed", 42),
        )
        obj.classes = tuple(data.get("classes", ["neg", "neu", "pos"]))  # type: ignore[assignment]
        obj.vocab_ = set(data.get("vocab", []))
        obj.log_likelihoods_ = {c: dict(tok_ll) for c, tok_ll in data["log_likelihoods"].items()}
        obj.tau_low_ = float(data["tau_low"])
        obj.tau_high_ = float(data["tau_high"])
        obj.dev_macro_f1_ = data.get("dev_macro_f1")
        return obj
