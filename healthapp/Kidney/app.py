from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from healthapp.web_security import create_legacy_app

HERE = Path(__file__).resolve().parent
PREDICTION_FEATURES = [
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
]

app = create_legacy_app(
    disease="Kidney",
    input_template="kidney.html",
    model_path=HERE / "final_models" / "kidney_model.pkl",
    preprocessor_path=HERE / "final_models" / "kidney_preprocessor.pkl",
    feature_names=PREDICTION_FEATURES,
    input_format="dataframe",
    high_risk_message="🔴 High Risk – please consult a nephrologist.",
    low_risk_message="🟢 Low Risk – maintain regular checkups!",
)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5004)
