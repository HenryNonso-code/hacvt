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
