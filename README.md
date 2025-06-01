
````markdown
# 🩺 Disease Predictor Web App

A **multi-disease prediction** web application that helps users assess the risk of **Heart Disease, Breast Cancer, Kidney Disease, Liver Disease**, and **Diabetes (for women)** using trained machine learning models. Built with Flask and deployed on Render.

🌐 **Live Demo:** [Disease Predictor](https://disease-predictor-dvty.onrender.com/)

---

## 🔍 Features

- 🔬 Predicts 5 major diseases: Heart, Breast Cancer, Kidney, Liver, Diabetes (for women).
- 📈 Risk output: `Low Risk` / `High Risk` based on user input.
- 💡 Separate prediction pages/tabs for each disease.
- 🧠 Trained and tracked 5 ML models using **MLflow**.
- 📦 Experiment tracking and model versioning via **DagsHub**.
- 💻 User interface built with **Flask**, **HTML/CSS**, and **Bootstrap**.
- 🚀 Deployed using **Render**.
- 🌐 Code managed with **GitHub**.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Flask (Python)
- **ML Libraries:** scikit-learn, pandas, NumPy
- **Model Tracking & Versioning:** MLflow, DagsHub
- **Version Control:** Git, GitHub
- **Deployment:** Render

---

## 🧪 Supported Diseases

| Disease         | Model Source    | Risk Output  |
|-----------------|------------------|--------------|
| Heart Disease   | Custom ML model  | Low / High   |
| Breast Cancer   | Custom ML model  | Low / High   |
| Kidney Disease  | Custom ML model  | Low / High   |
| Liver Disease   | Custom ML model  | Low / High   |
| Diabetes (Women)| Custom ML model  | Low / High   |

---

## 🔧 How It Works

1. User selects a disease tab from the disease section.
2. Fills in the medical form (e.g., age, BP, glucose).
3. Hits "Predict" to get the risk level (`Low Risk` or `High Risk`).
4. The model runs prediction in real-time using pre-trained models tracked via MLflow.

---

## 📦 Integration Details

- **MLflow:** Used to track experiments, parameters, metrics, and model versions.
- **DagsHub:** Hosted MLflow server and managed data/model versions.
- **GitHub:** Version control for full application code.

---

## 🏁 Getting Started Locally

```bash
git clone https://github.com/SamyakAnand/Disease-predictor.git
cd disease-predictor
pip install -r requirements.txt
python app.py
````

Then go to `http://127.0.0.1:5000/` in your browser.

---

## 📁 Project Structure

```
healthapp/
│
├── Heart/                 # Heart disease logic
├── Cancer/                # Breast cancer logic
├── Kidney/                # Kidney disease logic
├── Liver/                 # Liver disease logic
├── Diabetes/              # Diabetes prediction logic
├── templates/             # HTML UI
├── static/                # Images
├── app.py                 # Main Flask app
├── requirements.txt
└── README.md
```

---

## 📷 Screenshots

*Homepage:*
![Homepage](path_to_homepage_screenshot)

*Form Example:*
![Form](path_to_form_screenshot)

*Prediction Result:*
![Result](path_to_result_screenshot)

---

## 🚀 Future Improvements

* Add user authentication and prediction history.
* Allow CSV file input and batch predictions.
* Visualize confidence scores and SHAP values.
* Expand to include male-specific models and symptoms.

---

## 🙋‍♂️ Author

**Samyak Anand**
Aspiring Data Scientist | Hyderabad, India
[LinkedIn](https://www.linkedin.com/in/SamyakAnand/) | [GitHub](https://github.com/SamyakAnand) |

---

## 📄 License

Licensed under the **MIT License**. Free to use for personal, educational, and research purposes.
```


