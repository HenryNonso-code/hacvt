# examples/serve_api.py
# Run from project root:
# python -m uvicorn examples.serve_api:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path

from hacvt import HACVT

# ----------------------------
# Robust model loading
# ----------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "hacvt_model.json"

with MODEL_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

model = HACVT.from_dict(data)

app = FastAPI(title="HAC-VT Sentiment API")


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "HAC-VT sentiment API is running."}


@app.post("/predict")
def predict(input_data: TextInput):
    result = model.analyze(input_data.text)
    return {
        "text": result["text"],
        "label": result["label"],
        "delta": float(result["delta"]),
        "tau_low": float(result["tau_low"]),
        "tau_high": float(result["tau_high"]),
    }
