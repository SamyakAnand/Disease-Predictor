from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import pickle, json, os, uuid, datetime as dt
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# ─────────────────────── Global Configuration ───────────────────────
disease_configs = {
    "heart": {
         "MODEL_PATH": "healthapp/Heart/final_models/heart_model.pkl",
         "PREPROCESSOR_PATH": "healthapp/Heart/final_models/heart_preprocessor.pkl",
         "RESULTS_FILE": os.path.join("Results", "Heart", "results.txt"),
         "PREDICTION_FEATURES": ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang'],
         "INPUT_TEMPLATE": "heart.html",
         "RESULTS_TEMPLATE": "heart_results.html"
    },
    "diabetes": {
         "MODEL_PATH": "healthapp/Diabetes/final_models/diabetes_model.pkl",
         "PREPROCESSOR_PATH": "healthapp/Diabetes/final_models/diabetes_preprocessor.pkl",
         "RESULTS_FILE": os.path.join("Results", "Diabetes", "results.txt"),
         "PREDICTION_FEATURES": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
         "INPUT_TEMPLATE": "diabetes.html",
         "RESULTS_TEMPLATE": "diabetes_results.html"
    },
    "cancer": {
         "MODEL_PATH": "healthapp/Cancer/final_models/cancer_model.pkl",
         "PREPROCESSOR_PATH": "healthapp/Cancer/final_models/cancer_preprocessor.pkl",
         "RESULTS_FILE": os.path.join("Results", "Cancer", "results.txt"),
         "PREDICTION_FEATURES": ['concave points_mean', 'area_mean', 'radius_mean', 'perimeter_mean', 'concavity_mean'],
         "INPUT_TEMPLATE": "cancer.html",
         "RESULTS_TEMPLATE": "cancer_results.html"
    },
    "kidney": {
         "MODEL_PATH": "healthapp/Kidney/final_models/kidney_model.pkl",
         "PREPROCESSOR_PATH": "healthapp/Kidney/final_models/kidney_preprocessor.pkl",
         "RESULTS_FILE": os.path.join("Results", "Kidney", "results.txt"),
         # IMPORTANT: Update the kidney feature names to match those used at fit time.
         # In this example, we assume your training pipeline was fitted using these names:
         "PREDICTION_FEATURES": ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc'],
         "INPUT_TEMPLATE": "kidney.html",
         "RESULTS_TEMPLATE": "kidney_results.html"
    },
    "liver": {
         "MODEL_PATH": "healthapp/Liver/final_models/liver_model.pkl",
         "PREPROCESSOR_PATH": "healthapp/Liver/final_models/liver_preprocessor.pkl",
         "RESULTS_FILE": os.path.join("Results", "Liver", "results.txt"),
         "PREDICTION_FEATURES": ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'],
         "INPUT_TEMPLATE": "liver.html",
         "RESULTS_TEMPLATE": "liver_results.html"
    }
}

# ─────────────────────── Helper Functions ───────────────────────
def load_artifacts(disease: str):
    """Load model and preprocessor for the given disease."""
    try:
        cfg = disease_configs[disease]
        model = pickle.load(open(cfg["MODEL_PATH"], "rb"))
        preprocessor = pickle.load(open(cfg["PREPROCESSOR_PATH"], "rb"))
        print(f"✅ {disease.capitalize()} model & pre-processor loaded.")
        return model, preprocessor
    except Exception as e:
        print(f"❌ Couldn't load {disease} artefacts: {e}")
        return None, None

def _save_record(disease: str, rec: dict):
    """Append a record (as JSON) to the disease-specific results file."""
    cfg = disease_configs[disease]
    results_file = cfg["RESULTS_FILE"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec) + "\n")

def _load_records(disease: str) -> list:
    """Load stored records for the given disease."""
    cfg = disease_configs[disease]
    results_file = cfg["RESULTS_FILE"]
    if not os.path.exists(results_file):
        return []
    with open(results_file, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]

# ─────────────────────── Routes ───────────────────────
@app.route("/")
def home():
    # A simple home page that links to each disease's input form.
    return render_template("home.html", diseases=list(disease_configs.keys()))

@app.route("/<disease>")
def disease_page(disease):
    if disease not in disease_configs:
        return "Disease not supported", 404
    template = disease_configs[disease]["INPUT_TEMPLATE"]
    return render_template(template, disease=disease)

@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in disease_configs:
        return "Disease not supported", 404

    cfg = disease_configs[disease]
    try:
        f = request.form
        input_dict = {k: float(f.get(k, 0)) for k in cfg["PREDICTION_FEATURES"]}
        input_df = pd.DataFrame(input_dict, index=[0])
    except Exception as e:
        return render_template(cfg["INPUT_TEMPLATE"], prediction_text=f"❌ Error processing form: {e}", disease=disease)
    
    model, preprocessor = load_artifacts(disease)
    if model is None or preprocessor is None:
        return render_template(cfg["INPUT_TEMPLATE"], prediction_text=f"❌ Error loading {disease} model", disease=disease)
    
    try:
        x_t = preprocessor.transform(input_df)
        y_hat = model.predict(x_t)[0]
        msg = ("🔴 High Risk – please consult a doctor."
               if y_hat == 1 else
               "🟢 Low Risk – stay healthy!")
    except Exception as e:
        msg = f"❌ Error during prediction: {e}"
    
    rec_id = str(uuid.uuid4())
    record = {
        "id": rec_id,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        **{k: f.get(k, "") for k in cfg["PREDICTION_FEATURES"]},
        "Prediction": msg
    }
    _save_record(disease, record)
    return redirect(url_for("results", disease=disease, id=rec_id, view="current"))

@app.route("/results")
def results():
    disease = request.args.get("disease")
    rec_id = request.args.get("id")
    view = request.args.get("view", "all")
    if disease not in disease_configs:
        return "Disease not supported", 404

    all_recs = _load_records(disease)
    current = next((r for r in all_recs if r["id"] == rec_id), None)
    history = [r for r in all_recs if r["id"] != rec_id]
    if view == "current":
        history = []
    
    template = disease_configs[disease]["RESULTS_TEMPLATE"]
    return render_template(
        template,
        disease=disease,
        current_result=current,
        results=history,
        view_mode=view
    )

# ─────────────────────── Launch ───────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)