from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import pickle
import numpy as np
import json
import os

# Load trained models
MODEL_PATH = "healthapp/Diabetes/final_models/model.pkl"
PREPROCESSOR_PATH = "healthapp/Diabetes/final_models/diabetes_preprocessor.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))
    print("✅ Model and Preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/preprocessor: {e}")
    model, preprocessor = None, None

app = Flask(__name__)
CORS(app)

# Load past results function
def load_results():
    results = []
    if not os.path.exists("results.txt"):
        return results  # Return empty list if file doesn't exist
    
    with open("results.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            try:
                data_str = line.strip().split("Input: ")[1].split(", Prediction: ")[0]
                prediction = line.strip().split("Prediction: ")[1]
                data_dict = json.loads(data_str.replace("'", "\""))  # Convert string to dictionary
                data_dict["Prediction"] = prediction
                results.append(data_dict)
            except Exception as e:
                print(f"⚠ Error parsing results file: {e}")
    return results

@app.route("/")
def home():
    return render_template("diabetes.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = np.array([[float(data.get("Pregnancies", 0)), float(data.get("Glucose", 0)), float(data.get("BloodPressure", 0)),
                              float(data.get("SkinThickness", 0)), float(data.get("Insulin", 0)), float(data.get("BMI", 0)),
                              float(data.get("DiabetesPedigreeFunction", 0)), float(data.get("Age", 0))]])

        transformed_features = preprocessor.transform(features)
        prediction = model.predict(transformed_features)

        result_msg = "🔴 **High Risk**: Please consult a doctor for further screening." if prediction[0] == 1 else "🟢 **Low Risk**: Keep maintaining a healthy lifestyle!"

        # Save input values & result to file safely
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\nInput: {json.dumps(data.to_dict())}, Prediction: {result_msg}")

        return redirect(url_for("results"))  # Redirect to results page

    except Exception as e:
        return render_template("diabetes.html", prediction_text=f"❌ Error: {str(e)}")

@app.route("/results")
def results():
    past_results = load_results()
    return render_template("results.html", results=past_results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)