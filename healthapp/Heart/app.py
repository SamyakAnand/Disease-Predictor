# app.py
from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import pickle, json, os, uuid, datetime as dt
import numpy as np

# --------------------------------------------------------------------- #
# 1. Load model + pre-processor
# --------------------------------------------------------------------- #
MODEL_PATH        = "healthapp/Diabetes/final_models/diabetes_model.pkl"
PREPROCESSOR_PATH = "healthapp/Diabetes/final_models/diabetes_preprocessor.pkl"

try:
    model        = pickle.load(open(MODEL_PATH, "rb"))
    preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))
    print("✅  Model & pre-processor loaded!")
except Exception as e:
    print(f"❌  Couldn’t load model/pre-processor → {e}")
    model = preprocessor = None                       # fail-soft

# --------------------------------------------------------------------- #
# 2. Flask setup
# --------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app)

RESULTS_FILE = "results.txt"                          # one JSON per line


# --------------------------------------------------------------------- #
# 3. Helpers
# --------------------------------------------------------------------- #
def save_result(record: dict) -> None:
    """
    Persist a single screening (1 JSON / line).
    """
    with open(RESULTS_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")


def load_results() -> list[dict]:
    """
    Load ALL screenings (oldest → newest).
    If the file is missing / empty → returns [].
    """
    if not os.path.exists(RESULTS_FILE):
        return []

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print("⚠️  Skipping malformed line in results.txt")
    return records


# --------------------------------------------------------------------- #
# 4. Routes
# --------------------------------------------------------------------- #
@app.route("/")
def home():
    return render_template("diabetes.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    • Grab form fields
    • Transform + predict
    • Persist the record with a UID
    • Redirect to /results?id=<uid>&view=current
    """
    try:
        form = request.form

        feats = np.array([[float(form.get("Pregnancies", 0)),
                           float(form.get("Glucose", 0)),
                           float(form.get("BloodPressure", 0)),
                           float(form.get("SkinThickness", 0)),
                           float(form.get("Insulin", 0)),
                           float(form.get("BMI", 0)),
                           float(form.get("DiabetesPedigreeFunction", 0)),
                           float(form.get("Age", 0))]])

        # ‼️  sanity: do not crash if model is None
        if model is None or preprocessor is None:
            raise RuntimeError("Model or pre-processor not loaded.")

        x_t        = preprocessor.transform(feats)
        y_hat      = model.predict(x_t)[0]             # 0 = negative, 1 = positive

        result_msg = ("🔴  High Risk – please consult a doctor."
                      if y_hat == 1
                      else "🟢  Low Risk – keep up the healthy lifestyle!")

        # -------- persist -------- #
        record_id = str(uuid.uuid4())
        record = {
            "id":      record_id,
            "ts":      dt.datetime.utcnow().isoformat(),
            "Pregnancies":               form.get("Pregnancies", ""),
            "Glucose":                   form.get("Glucose", ""),
            "BloodPressure":             form.get("BloodPressure", ""),
            "SkinThickness":             form.get("SkinThickness", ""),
            "Insulin":                   form.get("Insulin", ""),
            "BMI":                       form.get("BMI", ""),
            "DiabetesPedigreeFunction":  form.get("DiabetesPedigreeFunction", ""),
            "Age":                       form.get("Age", ""),
            "Prediction":                result_msg
        }
        save_result(record)

        # Redirect to results page, showing just this screening first
        return redirect(url_for("results",
                                id=record_id,
                                view="current"))

    except Exception as e:
        return render_template("diabetes.html",
                               prediction_text=f"❌  Error: {e}")


@app.route("/results")
def results():
    """
    Query params:
      • id   – record UID to treat as “current screening”
      • view – 'current' | 'all'  (default = 'all')
    """
    record_id = request.args.get("id")
    view_mode = request.args.get("view", "all")

    all_records = load_results()
    current_rec = next((r for r in all_records if r["id"] == record_id), None)

    # remove current record from list so we don't duplicate in template
    past_records = [r for r in all_records if r["id"] != record_id]

    if view_mode == "current":
        past_records = []                              # hide history

    return render_template("results.html",
                           current_result=current_rec,
                           results=past_records,
                           view_mode=view_mode)


# --------------------------------------------------------------------- #
# 5. Launch
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)