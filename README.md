# 🎓 Student Risk Predictor

A machine learning–powered web application that predicts whether a student is at **academic risk** based on attendance, internal marks, study habits, and past performance. The goal is **early identification** so institutions can intervene before failure occurs.

🔗 **Live Demo:** [https://student-risk-predictor-09.streamlit.app/](https://student-risk-predictor-09.streamlit.app/)
🔗 **GitHub Repo:** [https://github.com/himanshu-jadhav108/student-risk-predictor](https://github.com/himanshu-jadhav108/student-risk-predictor)

---

## 📌 Problem Statement

Educational institutions often identify struggling students **too late**, after grades have already dropped. There is no simple, data‑driven tool to flag at‑risk students early using commonly available academic indicators.

---

## 💡 Proposed Solution

We built an **end‑to‑end ML system** that:

* Trains a predictive model using student academic data
* Estimates the probability of academic risk
* Provides instant predictions through a web interface
* Is fully deployable and reproducible

The system helps faculty and mentors take **preventive action** instead of reactive measures.

---

## 🧠 Machine Learning Approach

### Model Used

* **Logistic Regression**

### Why Logistic Regression?

* Works well on small to medium tabular datasets
* Fast to train and deploy
* Interpretable (important for education domain)
* Probability‑based output (risk score)

---

## 📊 Dataset Description

**Features:**

* `attendance` – Percentage of classes attended
* `internal_marks` – Internal assessment score
* `study_hours` – Average daily study time
* `assignments_completed` – Number of assignments completed
* `previous_failures` – Count of previous failures

**Target:**

* `risk` → 0 = Low Risk, 1 = High Risk

**Dataset Size:** Prototype dataset (can scale to real institutional data)

---

## ⚙️ Model Training & Evaluation

### Train / Test Split

* 80% Training
* 20% Testing

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1‑Score

> Metrics are printed during training and help validate model reliability.

---

## 📈 Visualizations Included

The application includes:

* Feature distribution charts
* Risk probability visualization
* Model decision explanation (optional)

Graphs are displayed **only after prediction**, ensuring a clean UI.

---

## 🖥️ Tech Stack

| Layer         | Technology                |
| ------------- | ------------------------- |
| Frontend      | Streamlit                 |
| Backend       | Python                    |
| ML            | scikit‑learn              |
| Data          | Pandas, NumPy             |
| Visualization | Matplotlib                |
| Deployment    | Streamlit Community Cloud |

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Auto‑Update Behavior

* Any push to the `main` branch automatically updates the live app
* No manual redeployment required

---

## ▶️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/himanshu-jadhav108/student-risk-predictor.git
cd student-risk-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run app
streamlit run app.py
```

---

## 🧪 Example Workflow

1. User inputs student details
2. Model predicts risk probability
3. UI displays:

   * Risk category
   * Confidence score
   * Supporting graphs

---

## 🔍 Reproducibility & Transparency

* All code available on GitHub
* Dataset included
* Model retraining supported
* No paid APIs or services used

---

## 🔮 Future Enhancements

* Larger real‑world dataset integration
* Advanced models (Random Forest, XGBoost)
* SHAP‑based explainability
* User authentication (admin/faculty)
* Student‑wise historical tracking

---

## 🏁 Conclusion

This project demonstrates a **complete ML lifecycle** — from data to deployment — using free and open‑source tools. It is lightweight, explainable, and scalable, making it ideal for academic environments.

---

## 👨‍💻 About the Maintainer

**Himanshu Jadhav**  
Second-Year Engineering Student (AI & Data Science)

### Connect with me:

[![GitHub](https://img.shields.io/badge/GitHub-himanshu--jadhav108-black?style=flat-square&logo=github)](https://github.com/himanshu-jadhav108)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-himanshu--jadhav-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/himanshu-jadhav-328082339?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
[![Instagram](https://img.shields.io/badge/Instagram-himanshu__jadhav__108-purple?style=flat-square&logo=instagram)](https://www.instagram.com/himanshu_jadhav_108?igsh=MWYxamppcTBlY3Rl)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Now-yellow?style=flat-square)](https://himanshu-jadhav-portfolio.vercel.app/)

---
