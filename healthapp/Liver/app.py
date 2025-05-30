from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import pickle, json, os, uuid, datetime as dt, numpy as np
import pandas as pd

# ──────────────────────── 1. Load ML Model ────────────────────────
MODEL_PATH = "healthapp/Liver/final_models/liver_model.pkl"
PREPROCESSOR_PATH = "healthapp/Liver/final_models/liver_preprocessor.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))
    print("✅ Liver model & pre-processor loaded.")
except Exception as e:
    print(f"❌ Couldn’t load artifacts → {e}")
    model = preprocessor = None

# ───────────────────────── 2. Flask setup ──────────────────────────
app = Flask(__name__)
CORS(app)

RESULTS_FILE = "healthapp/Liver/results.txt"

def _save(rec: dict):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec) + "\n")

def _load() -> list[dict]:
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]

FEATURE_KEYS = [
    "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Total_Protiens", "Albumin",
    "Albumin_and_Globulin_Ratio"
]

# ─────────────────────────── 3. Routes ─────────────────────────────
@app.route("/")
def home():
    return render_template("liver.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form
        input_df = pd.DataFrame({k: [float(f.get(k, 0))] for k in FEATURE_KEYS})

        if model is None or preprocessor is None:
            raise RuntimeError("Model not loaded")

        x_t = preprocessor.transform(input_df)
        y_hat = model.predict(x_t)[0]

        msg = ("🔴 High Risk – please consult a hepatologist."
               if y_hat == 1 else
               "🟢 Low Risk – keep monitoring regularly.")

        rec_id = str(uuid.uuid4())
        record = {
            "id": rec_id,
            "ts": dt.datetime.utcnow().isoformat(),
            **{k: f.get(k, "") for k in FEATURE_KEYS},
            "Prediction": msg
        }
        _save(record)

        return redirect(url_for("results", id=rec_id, view="current"))

    except Exception as e:
        return render_template("liver.html", prediction_text=f"❌ Error: {e}")

@app.route("/results")
def results():
    rec_id = request.args.get("id")
    view = request.args.get("view", "all")

    all_recs = _load()
    current = next((r for r in all_recs if r["id"] == rec_id), None)
    history = [r for r in all_recs if r["id"] != rec_id]

    if view == "current":
        history = []

    return render_template("liver_results.html", current_result=current, results=history, view_mode=view)

# ────────────────────────── 4. Launch ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)