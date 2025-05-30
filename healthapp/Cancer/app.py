from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import pickle, json, os, uuid, datetime as dt, numpy as np
import pandas as pd  # Needed for DataFrame creation

# ──────────────────────── 1. Load artefacts ────────────────────────
MODEL_PATH        = "healthapp/Cancer/final_models/cancer_model.pkl"
PREPROCESSOR_PATH = "healthapp/Cancer/final_models/cancer_preprocessor.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))
    print("✅ Cancer model & pre-processor loaded.")
except Exception as e:
    print(f"❌ Couldn’t load artefacts → {e}")
    model = preprocessor = None

# ───────────────────────── 2. Flask setup ──────────────────────────
app = Flask(__name__)
CORS(app)

RESULTS_FILE = os.path.join("healthapp", "Cancer", "results.txt")

def _save(rec: dict):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec) + "\n")

def _load() -> list[dict]:
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]

# Use exactly the same features for prediction as were used during training.
PREDICTION_FEATURES = [
    "concave points_mean", "area_mean", "radius_mean", "perimeter_mean", "concavity_mean"
]

# ─────────────────────────── 3. Routes ─────────────────────────────
@app.route("/")
def home():
    return render_template("cancer.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form
        # Build a DataFrame with feature names matching those used in training
        input_df = pd.DataFrame({k: [float(f.get(k, 0))] for k in PREDICTION_FEATURES})
        
        if model is None or preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded")
        
        # Preprocessor was fitted on a DataFrame having columns PREDICTION_FEATURES.
        x_t = preprocessor.transform(input_df)
        y_hat = model.predict(x_t)[0]  # 1 indicates malignant (cancer risk)
        
        msg = ("🔴 High Risk – please consult an oncologist."
               if y_hat == 1 else
               "🟢 Low Risk – maintain regular screenings!")
        
        rec_id = str(uuid.uuid4())
        # Save only the prediction features (plus timestamp and prediction message)
        record = {
            "id": rec_id,
            "ts": dt.datetime.utcnow().isoformat(),
            **{k: f.get(k, "") for k in PREDICTION_FEATURES},
            "Prediction": msg
        }
        _save(record)
        
        return redirect(url_for("results", id=rec_id, view="current"))
    
    except Exception as e:
        return render_template("cancer.html", prediction_text=f"❌ Error: {e}")

@app.route("/results")
def results():
    rec_id = request.args.get("id")
    view = request.args.get("view", "all")
    
    all_recs = _load()
    current = next((r for r in all_recs if r["id"] == rec_id), None)
    history = [r for r in all_recs if r["id"] != rec_id]
    
    if view == "current":
        history = []
    
    return render_template("cancer_results.html",
                           current_result=current,
                           results=history,
                           view_mode=view)

# ────────────────────────── 4. Launch ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)