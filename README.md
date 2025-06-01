# 🧠 Disease Predictor: Multi-Disease ML System (Modular & Scalable)

A robust, modular, and production-ready **machine learning system** that predicts the risk of **five major diseases**:

- ❤️ Heart Disease
- 🎗️ Breast Cancer
- 🧪 Kidney Disease
- 🧬 Liver Disease
- 🍭 Diabetes (for women)

Each disease is independently supported with a complete ML lifecycle, including **data ingestion, validation, transformation, training**, and **drift detection**. Built using Flask and deployed on Render with full experiment tracking and model versioning.

🌐 **Live App:** [Try it on Render](https://disease-predictor-dvty.onrender.com/)

---

## ✨ Key Highlights

- 🔁 **Modular architecture** for each disease
- 🧪 Custom ML pipelines: ingestion → validation → transformation → training
- 🧬 Schema validation and data drift detection
- 📈 **MLflow** for experiment tracking
- 📦 Version control for model/data via **DagsHub**
- 🧾 Clean and responsive UI (HTML, CSS, Bootstrap)
- 🌍 RESTful backend with Flask, deployed via **Waitress + Render**
- 📊 Logging & error tracking with custom exception handling

---

## 🏗️ Project Architecture

```
healthapp/
├── Cancer/         # Full pipeline and logic for breast cancer prediction
│   ├── components/ # ingestion, validation, transformation, training
│   ├── utils/      # metric calculators, estimators
│   └── app.py      # Flask route
├── Diabetes/
├── Heart/
├── Kidney/
├── Liver/
├── app_logging/    # Logger utility
├── exception/      # Custom exception classes
├── main.py         # Main training/inference entrypoint
├── templates/      # HTML templates
├── static/         # CSS, images
└── ...
```

Each disease directory includes:
- `data_ingestion.py`
- `data_validation.py` (+ drift report)
- `data_transformation.py`
- `model_trainer.py`
- `schema.yaml`, constants, config entities

---

## 🔧 Tech Stack

| Category             | Technologies Used                                      |
|----------------------|--------------------------------------------------------|
| **Frontend**         | HTML, CSS, Bootstrap                                   |
| **Backend**          | Flask, Flask-CORS, Waitress                            |
| **ML Libraries**     | scikit-learn, pandas, NumPy, matplotlib                |
| **Pipeline Modules** | Modular Python (custom components per disease)         |
| **Tracking**         | MLflow, DagsHub, PyYAML                                |
| **Database**         | MongoDB (via PyMongo), certifi                         |
| **ETL Pipeline**     | Python with schema validation + drift detection        |
| **Deployment**       | Render (WSGI via Waitress)                             |
| **Utilities**        | python-dotenv, logging, exception handling             |

---

## 🚀 Quick Start (Local Development)

```bash
git clone https://github.com/SamyakAnand/Disease-predictor.git
cd disease-predictor
pip install -r requirements.txt
python healthapp/main.py
```

Access via browser at: `http://127.0.0.1:5000/`

---

## 🧠 Supported Diseases & Models

| Disease          | Pipeline             | Output     |
|------------------|----------------------|------------|
| Heart Disease    | Custom ML pipeline   | Low/High   |
| Breast Cancer    | Custom ML pipeline   | Low/High   |
| Kidney Disease   | Custom ML pipeline   | Low/High   |
| Liver Disease    | Custom ML pipeline   | Low/High   |
| Diabetes (Women) | Custom ML pipeline   | Low/High   |

---

## 🔗 ML System Integration

- **ETL pipeline**: Extract from MongoDB → Validate (schema + drift) → Transform → Train
- **MLflow & DagsHub**: Model tracking, artifact versioning
- **Modular Design**: Each disease is a fully independent, pluggable module

---

## 🗃️ Screenshots

**🔹 Homepage**
![Homepage](https://github.com/SamyakAnand/Disease-Predictor/blob/main/images/Home.png)

**🔹 Prediction Form**
![Form](https://github.com/SamyakAnand/Disease-Predictor/blob/main/images/Predictor.png)

**🔹 Prediction Output**
![Result](https://github.com/SamyakAnand/Disease-Predictor/blob/main/images/Result.png)

---

## 📈 Future Enhancements

- 👥 User authentication & prediction history
- 📁 CSV upload for batch prediction
- 🔍 Model explainability (SHAP, feature importance)
- 🧔 Models for male-specific conditions
- 📊 Monitoring dashboards

---

## 👨‍💻 Author

**Samyak Anand**  
Aspiring Data Scientist | Hyderabad, India  
[🔗 LinkedIn](https://www.linkedin.com/in/SamyakAnand/)  
[💻 GitHub](https://github.com/SamyakAnand)

---

## 📝 License

Distributed under the **MIT License** – free for personal, academic, and commercial use with attribution.


