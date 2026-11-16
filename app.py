from flask import Flask, render_template, request
import datetime as dt
import os
import pickle

import numpy as np
import pandas as pd

app = Flask(__name__)

# ─────────────────────── Global Configuration ───────────────────────
disease_configs = {
    "heart": {
        "MODEL_PATH": "healthapp/Heart/final_models/heart_model.pkl",
        "PREPROCESSOR_PATH": "healthapp/Heart/final_models/heart_preprocessor.pkl",
        "PREDICTION_FEATURES": ["cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang"],
        "INPUT_TEMPLATE": "heart.html",
        "RESULTS_TEMPLATE": "heart_results.html",
    },
    "diabetes": {
        "MODEL_PATH": "healthapp/Diabetes/final_models/diabetes_model.pkl",
        "PREPROCESSOR_PATH": "healthapp/Diabetes/final_models/diabetes_preprocessor.pkl",
        "PREDICTION_FEATURES": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ],
        "INPUT_TEMPLATE": "diabetes.html",
        "RESULTS_TEMPLATE": "diabetes_results.html",
    },
    "cancer": {
        "MODEL_PATH": "healthapp/Cancer/final_models/cancer_model.pkl",
        "PREPROCESSOR_PATH": "healthapp/Cancer/final_models/cancer_preprocessor.pkl",
        "PREDICTION_FEATURES": [
            "concave points_mean", "area_mean", "radius_mean",
            "perimeter_mean", "concavity_mean",
        ],
        "INPUT_TEMPLATE": "cancer.html",
        "RESULTS_TEMPLATE": "cancer_results.html",
    },
    "kidney": {
        "MODEL_PATH": "healthapp/Kidney/final_models/kidney_model.pkl",
        "PREPROCESSOR_PATH": "healthapp/Kidney/final_models/kidney_preprocessor.pkl",
        "PREDICTION_FEATURES": ["bp", "sg", "al", "su", "rbc", "pc", "pcc"],
        "INPUT_TEMPLATE": "kidney.html",
        "RESULTS_TEMPLATE": "kidney_results.html",
    },
    "liver": {
        "MODEL_PATH": "healthapp/Liver/final_models/liver_model.pkl",
        "PREPROCESSOR_PATH": "healthapp/Liver/final_models/liver_preprocessor.pkl",
        "PREDICTION_FEATURES": [
            "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
            "Alamine_Aminotransferase", "Total_Protiens", "Albumin",
            "Albumin_and_Globulin_Ratio",
        ],
        "INPUT_TEMPLATE": "liver.html",
        "RESULTS_TEMPLATE": "liver_results.html",
    },
}


def load_artifacts(disease: str):
    """Load the model and preprocessor for one supported disease."""
    config = disease_configs[disease]
    try:
        with open(config["MODEL_PATH"], "rb") as model_file:
            model = pickle.load(model_file)
        with open(config["PREPROCESSOR_PATH"], "rb") as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
        return model, preprocessor
    except Exception:
        app.logger.exception("Could not load model artifacts for disease=%s", disease)
        return None, None


def _build_input_frame(form, feature_names):
    """Require every configured feature and reject non-numeric/non-finite values."""
    values = {}
    for feature in feature_names:
        raw_value = form.get(feature)
        if raw_value is None or not raw_value.strip():
            raise ValueError("Please complete every field before submitting.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError("Enter a valid number for every field.") from None

        if not np.isfinite(value):
            raise ValueError("Values must be finite numbers.")

        values[feature] = value

    return values, pd.DataFrame(values, index=[0])


@app.after_request
def prevent_sensitive_response_caching(response):
    """Avoid storing screening pages in shared or browser caches."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# ─────────────────────── Routes ───────────────────────
@app.route("/")
def home():
    return render_template("home.html", diseases=list(disease_configs.keys()))


@app.route("/<disease>")
def disease_page(disease):
    if disease not in disease_configs:
        return "Disease not supported", 404
    return render_template(disease_configs[disease]["INPUT_TEMPLATE"], disease=disease)


@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in disease_configs:
        return "Disease not supported", 404

    config = disease_configs[disease]
    try:
        input_values, input_frame = _build_input_frame(
            request.form, config["PREDICTION_FEATURES"]
        )
    except ValueError as error:
        return (
            render_template(
                config["INPUT_TEMPLATE"],
                prediction_text=str(error),
                disease=disease,
            ),
            400,
        )

    model, preprocessor = load_artifacts(disease)
    if model is None or preprocessor is None:
        return (
            render_template(
                config["INPUT_TEMPLATE"],
                prediction_text="Prediction service is temporarily unavailable. Please try again later.",
                disease=disease,
            ),
            503,
        )

    try:
        transformed = preprocessor.transform(input_frame)
        prediction = model.predict(transformed)[0]
    except Exception:
        app.logger.exception("Prediction failed for disease=%s", disease)
        return (
            render_template(
                config["INPUT_TEMPLATE"],
                prediction_text="The prediction could not be completed. Please try again later.",
                disease=disease,
            ),
            500,
        )

    message = (
        "🔴 High Risk – please consult a qualified healthcare professional."
        if prediction == 1
        else "🟢 Low Risk – this model output is not a medical assessment."
    )
    current_result = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        **input_values,
        "Prediction": message,
    }

    # The public demo intentionally does not persist health inputs or expose
    # a shared prediction-history view. Render only this request's result.
    return render_template(
        config["RESULTS_TEMPLATE"],
        disease=disease,
        current_result=current_result,
        results=[],
        view_mode="current",
    )


@app.route("/results")
def results():
    return (
        "Prediction history is disabled in this demo. Submit a new screening to view a result.",
        410,
    )


# ─────────────────────── Launch ───────────────────────
if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
