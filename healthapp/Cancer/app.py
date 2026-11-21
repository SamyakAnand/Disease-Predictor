from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from healthapp.web_security import create_legacy_app

HERE = Path(__file__).resolve().parent
PREDICTION_FEATURES = [
    "concave points_mean",
    "area_mean",
    "radius_mean",
    "perimeter_mean",
    "concavity_mean",
]

app = create_legacy_app(
    disease="Cancer",
    input_template="cancer.html",
    model_path=HERE / "final_models" / "cancer_model.pkl",
    preprocessor_path=HERE / "final_models" / "cancer_preprocessor.pkl",
    feature_names=PREDICTION_FEATURES,
    input_format="dataframe",
    high_risk_message="🔴 High Risk – please consult an oncologist.",
    low_risk_message="🟢 Low Risk – maintain regular screenings!",
)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5003)
