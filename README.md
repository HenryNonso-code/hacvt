from hacvt import HACVT
import json
from pathlib import Path

# Path to the pretrained HAC-VT model (author-provided or learned)
MODEL_PATH = Path("hacvt_model.json")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "hacvt_model.json not found. Please ensure the pretrained model file "
        "is present in the current directory."
    )

# Load pretrained HAC-VT model
with MODEL_PATH.open("r", encoding="utf-8") as f:
    model_data = json.load(f)

model = HACVT.from_dict(model_data)

# Label-only prediction
print(model.predict_one("The car is good, not terrible."))

# Full analysis with calibrated neutral band and Δ-score
print(model.analyze("I am not happy with this engine at all."))


---

## Profile JSON Schema (Canonical)

HAC-VT uses a **profile.json** file as the single source of truth for
calibration, evaluation, API inference, and dashboard interaction.

### Required fields

```json
{
  "name": "string",
  "version": "1.0",
  "tau": 0.18,
  "label_map": { "neg": 0, "neu": 1, "pos": 2 },
  "params": {},
  "meta": {},
  "calibration_report": {}
}

Field definitions

name: Human-readable profile name (e.g. dataset + date)

version: Profile schema version

tau: Calibrated neutral-band threshold (required for dataset evaluation)

label_map: Mapping from sentiment labels to numeric IDs

params: HAC-VT scoring parameters

meta: Dataset metadata and provenance

calibration_report: Calibration metrics and diagnostics

Example profile.json
{
  "name": "car_reviews_hacvt_2025_12_25",
  "version": "1.0",
  "tau": 0.18,
  "label_map": { "neg": 0, "neu": 1, "pos": 2 },
  "params": {
    "tokenizer": {
      "lowercase": true,
      "keep_punct": false,
      "negation_scope": 3
    },
    "scoring": {
      "decision_rule": "pos_ll_minus_neg_ll"
    }
  },
  "meta": {
    "dataset": "CarReviews",
    "timestamp_utc": "2025-12-25T05:30:00Z",
    "text_col": "text",
    "label_col": "label",
    "n_dev": 2000,
    "n_test": 5000
  },
  "calibration_report": {
    "metric": "macro_f1",
    "dev_macro_f1": 0.53,
    "tau_grid_size": 101,
    "class_counts_dev": { "neg": 800, "neu": 800, "pos": 400 }
  }
}

Mandatory workflow

Calibrate tau on a labelled development set

Save the resulting profile.json

Evaluate datasets using the calibrated profile

Dataset evaluation without a calibrated tau is not allowed.
