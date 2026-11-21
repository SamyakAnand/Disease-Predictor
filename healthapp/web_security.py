"""Shared safety helpers for the standalone legacy disease-prediction demos."""

import math
import pickle
from pathlib import Path

from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InvalidFormData(ValueError):
    """Raised when a required numeric prediction field is missing or invalid."""


def load_pickle(path):
    """Load a trusted, repository-managed artifact and always close its file."""
    with Path(path).open("rb") as artifact_file:
        return pickle.load(artifact_file)


def parse_numeric_form(form, feature_names):
    """Return required finite numeric inputs in the model's expected feature order."""
    values = {}
    for name in feature_names:
        raw_value = form.get(name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise InvalidFormData("Every prediction field is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise InvalidFormData("Every prediction field must be numeric.") from exc

        if not math.isfinite(value):
            raise InvalidFormData("Prediction fields must contain finite numbers.")

        values[name] = value

    return values


def create_legacy_app(
    *,
    disease,
    input_template,
    model_path,
    preprocessor_path,
    feature_names,
    input_format,
    high_risk_message,
    low_risk_message,
):
    """Build an isolated legacy demo without shared screening-history storage."""
    if input_format not in {"dataframe", "array"}:
        raise ValueError("input_format must be 'dataframe' or 'array'.")

    app = Flask(
        f"legacy_{disease.lower()}",
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    try:
        model = load_pickle(model_path)
        preprocessor = load_pickle(preprocessor_path)
    except Exception:
        app.logger.exception("Could not load model artifacts for %s", disease)
        model = preprocessor = None

    app.config.update(
        DEBUG=False,
        MODEL=model,
        PREPROCESSOR=preprocessor,
        FEATURE_NAMES=tuple(feature_names),
        DISEASE=disease,
        INPUT_TEMPLATE=input_template,
    )

    @app.after_request
    def disable_response_caching(response):
        response.headers["Cache-Control"] = "private, no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    def render_form(message=None, status=200):
        return (
            render_template(
                input_template,
                disease=disease,
                prediction_text=message,
            ),
            status,
        )

    @app.route("/")
    def home():
        return render_template(input_template, disease=disease)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            values = parse_numeric_form(
                request.form,
                app.config["FEATURE_NAMES"],
            )
        except InvalidFormData:
            return render_form(
                "Please enter a valid finite number for every required field.",
                400,
            )

        model_instance = app.config["MODEL"]
        preprocessor_instance = app.config["PREPROCESSOR"]
        if model_instance is None or preprocessor_instance is None:
            app.logger.error("Prediction artifacts are unavailable for %s", disease)
            return render_form(
                "Prediction is temporarily unavailable. Please try again later.",
                503,
            )

        try:
            if input_format == "dataframe":
                import pandas as pd

                features = pd.DataFrame(
                    [values],
                    columns=app.config["FEATURE_NAMES"],
                )
            else:
                import numpy as np

                features = np.asarray(
                    [[values[name] for name in app.config["FEATURE_NAMES"]]],
                    dtype=float,
                )

            transformed = preprocessor_instance.transform(features)
            prediction = model_instance.predict(transformed)[0]
            message = (
                high_risk_message
                if prediction == 1
                else low_risk_message
            )
            return render_form(message)
        except Exception:
            app.logger.exception("Prediction failed for %s", disease)
            return render_form(
                "Prediction could not be completed. Please try again later.",
                503,
            )

    @app.route("/results")
    def results():
        return (
            "Shared screening history is disabled to protect privacy.",
            410,
        )

    return app
