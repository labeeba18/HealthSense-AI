# HealthSense-AI
AI-based health monitoring web app


# 🧠 Disease Prediction System

A machine learning-based web application that predicts diseases (such as diabetes) using user input data. The system is built with multiple trained models and deployed using Streamlit to provide an interactive and user-friendly interface.

---


## 📌 Features

* 🤖 Multiple ML models (Logistic Regression, Decision Tree, Random Forest)
* 📊 Real-time disease prediction based on user inputs
* 🧠 Pre-trained models for accurate results
* 📁 Structured data pipeline and preprocessing
* 💾 Integrated database for storing patient data
* 🖥️ Interactive UI using Streamlit

---

## 🛠️ Technologies Used

* **Frontend/UI:** Streamlit
* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Handling:** NumPy, Pandas
* **Database:** SQLite
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
Disease Prediction System/
│── assets/
│── data/
│   ├── diabetes.csv
│   ├── X_train_scaled.npy
│   ├── X_test_scaled.npy
│   ├── y_train.npy
│   ├── y_test.npy
│
│── models/
│   ├── Logistic_Regression_model.pkl
│   ├── Decision_Tree_model.pkl
│   ├── Random_Forest_model.pkl
│   ├── scaler.pkl
│   ├── evaluation_results.json
│
│── src/
│   ├── data_pipeline.py
│   ├── database.py
│   ├── train_models.py
│
│── app.py
│── patients.db
│── requirements.txt
│── README.md
```

---

## ⚙️ How It Works

1. User enters health-related inputs
2. Data is preprocessed using saved scaler
3. Multiple ML models predict the outcome
4. Best prediction is displayed via Streamlit UI
5. Data can be stored in the database for future use

---

## 🚀 Running the Project

1. Clone the repository

```
git clone https://github.com/your-username/disease-prediction-system.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
streamlit run app.py
```

---

## 🎯 Purpose

* To predict diseases using machine learning
* To demonstrate end-to-end ML pipeline
* To combine data science with a real-world web interface

---

## 🧠 Future Scope

* Add more disease prediction models
* Improve accuracy with advanced algorithms
* Deploy on cloud (Streamlit Cloud / AWS)
* Add user login and history tracking

---

## 👩‍💻 Author

**Labeeba Aafeen**
Data Science Student
I & Web Development
