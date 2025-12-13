from hacvt import HACVT
import json

# Load pre-trained model provided by author
with open("hacvt_model.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = HACVT.from_dict(data)

print(model.predict_one("The car is good, not terrible."))
print(model.analyze("I am not happy with this engine at all."))
