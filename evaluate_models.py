"""
evaluate_models.py — HAC-VT standalone evaluation (Step A.2)

Goal of this step:
- Confirm the hacvt package imports correctly
- Locate key entry points for prediction (HAC-VT + baselines)
- Do NOT run training/calibration
- Do NOT require your full notebook pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    # This file lives at repo root, so root is its parent directory
    return Path(__file__).resolve().parent


def _add_repo_to_path():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _try_imports():
    """
    Try to import the hacvt package and common modules.
    We intentionally keep this flexible because repo layouts vary slightly.
    """
    results = {}

    # 1) Core package import
    try:
        import hacvt  # noqa: F401
        results["hacvt (package)"] = "OK"
    except Exception as e:
        results["hacvt (package)"] = f"FAIL: {type(e).__name__}: {e}"

    # 2) Try likely HAC-VT model entry points (we will use whichever exists)
    candidates = [
        "hacvt.model",
        "hacvt.models",
        "hacvt.core",
        "hacvt.hacvt_core",
        "hacvt.predict",
        "hacvt.inference",
    ]
    for mod in candidates:
        try:
            __import__(mod)
            results[mod] = "OK"
        except Exception as e:
            results[mod] = f"FAIL: {type(e).__name__}: {e}"

    # 3) Try likely baseline modules (VADER/TextBlob wrappers)
    baseline_candidates = [
        "hacvt.baselines",
        "hacvt.baseline",
        "hacvt.vader",
        "hacvt.textblob",
        "hacvt.utils",
        "hacvt.eval",
        "hacvt.evaluation",
    ]
    for mod in baseline_candidates:
        try:
            __import__(mod)
            results[mod] = "OK"
        except Exception as e:
            results[mod] = f"FAIL: {type(e).__name__}: {e}"

    return results


def main():
    _add_repo_to_path()

    print("=== HAC-VT Standalone Evaluation — Step A.2 ===")
    print(f"Repo root: {_repo_root()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    results = _try_imports()

    # Print summary neatly
    ok = [k for k, v in results.items() if v == "OK"]
    fail = {k: v for k, v in results.items() if v != "OK"}

    print("---- Import checks (OK) ----")
    for k in ok:
        print(f"  ✅ {k}")

    print("\n---- Import checks (FAIL) ----")
    if not fail:
        print("  (none)")
    else:
        for k, v in fail.items():
            print(f"  ❌ {k} -> {v}")

    print("\nNext step:")
    print("- If 'hacvt (package)' is OK, we proceed to Step A.3 (load test CSV + wire predict functions).")
    print("- If it FAILS, run:  pip install -e .   from the repo root, then re-run this script.")


if __name__ == "__main__":
    main()
