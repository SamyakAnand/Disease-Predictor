from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from healthapp.web_security import create_legacy_app

HERE = Path(__file__).resolve().parent
PREDICTION_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

app = create_legacy_app(
    disease="Diabetes",
    input_template="diabetes.html",
    model_path=HERE / "final_models" / "diabetes_model.pkl",
    preprocessor_path=HERE / "final_models" / "diabetes_preprocessor.pkl",
    feature_names=PREDICTION_FEATURES,
    input_format="array",
    high_risk_message="🔴 High Risk – please consult a doctor.",
    low_risk_message="🟢 Low Risk – keep up the healthy lifestyle!",
)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
