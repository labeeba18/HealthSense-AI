# HealthSense AI (Enterprise Disease Prediction System)

This is a complete final year project showcasing a robust Machine Learning pipeline for predicting diseases (using the Pima Indians Diabetes dataset as the primary example). It features data preprocessing, classification models (Logistic Regression, Decision Tree, Random Forest), an SQLite database for history, and a modern, professional, dark-themed Streamlit web application.

## 📂 Project Structure

```text
Disease Prediction System/
├── data/                       # Contains raw and preprocessed datasets
├── models/                     # Saved models (.pkl) and evaluation results (.json)
├── src/                        # Source python scripts
│   ├── data_pipeline.py        # Downloads, cleans, and scales data
│   ├── train_models.py         # Trains models and generates evaluation metrics
│   └── database.py             # SQLite database logic
├── app.py                      # Main Streamlit web application
├── patients.db                 # SQLite DB storing patient prediction history
├── requirements.txt            # Python dependencies
├── README.md                   # Setup details
└── project_report.md           # Report for Viva and final documentation
```

## 🚀 How to Run the Project

### 1. Install Dependencies
Open your terminal/command prompt and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
This script downloads the dataset, handles missing values, scales it using `StandardScaler`, and prepares train/test splits.
```bash
python src/data_pipeline.py
```

### 3. Train the Models
Train Logistic Regression, Decision Tree, and Random Forest models. This script also generates prediction metrics and saves them.
```bash
python src/train_models.py
```

### 4. Start the Application
Initialize the Streamlit Web Application using Python:
```bash
py -m streamlit run app.py
```
This will automatically open the web application in your default browser.

## ✨ Features
- **Enterprise-Grade UI**: A completely custom, dark-mode inspired professional healthcare interface with a fully responsive card-based layout and smooth animations.
- **AI Detection**: Predict if a patient is at risk of diabetes using top-tier ML algorithms with immediate, actionable AI Diagnosis Reports.
- **Patient History Database**: Save and manage all predictions automatically via an internal SQLite database (`patients.db`).
- **Analytics Dashboard**: Monitor real-time charts (Plotly) and global health metrics securely formatted within the application.
