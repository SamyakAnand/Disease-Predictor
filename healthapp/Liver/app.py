from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from healthapp.web_security import create_legacy_app

HERE = Path(__file__).resolve().parent
PREDICTION_FEATURES = [
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]

app = create_legacy_app(
    disease="Liver",
    input_template="liver.html",
    model_path=HERE / "final_models" / "liver_model.pkl",
    preprocessor_path=HERE / "final_models" / "liver_preprocessor.pkl",
    feature_names=PREDICTION_FEATURES,
    input_format="dataframe",
    high_risk_message="🔴 High Risk – please consult a hepatologist.",
    low_risk_message="🟢 Low Risk – keep monitoring regularly.",
)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5002)
