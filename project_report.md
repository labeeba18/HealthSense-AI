# Final Year Project Report: Disease Prediction System Using Machine Learning

## 1. Project Overview
This project focuses on leveraging Machine Learning techniques to predict whether a patient is at risk of diabetes based on several physiological metrics. The project integrates a complete data engineering pipeline, a set of classification models, database connectivity, and a modern frontend interface.

## 2. Methodology & Implementation Details (Viva Explanation)

### Step 1: Data Acquisition & Preprocessing (`src/data_pipeline.py`)
- **Dataset**: We utilized the Pima Indians Diabetes Dataset. It includes features like Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
- **Data Cleaning**: One of the primary issues with medical datasets is missing values masked as `0` (e.g., Blood Pressure = 0). We addressed this by writing a pipeline that replaces `0`s with `NaN` for appropriate columns, and then fills them using the respective column **mean**.
- **Scaling**: We used `scikit-learn`'s `StandardScaler` to ensure all features contribute equally to the model by removing the mean and scaling to unit variance.

### Step 2: Model Building (`src/train_models.py`)
- **Logistic Regression**: Used as the baseline model. It applies a sigmoid function to a linear equation to find probabilities of classification.
- **Decision Tree Classifier**: Maps decisions in a tree structure. Captures non-linear relationships well, but can be prone to overfitting.
- **Random Forest Classifier**: An ensemble method that combines multiple decision trees. This usually provides the highest accuracy and generalizability for tabular data like ours, significantly reducing the probability of overfitting.

### Step 3: Model Evaluation
The models were evaluated using four key metrics:
1. **Accuracy**: Total correct predictions over total predictions.
2. **Precision**: Out of all the patients the model flagged as 'Diabetic', how many were actually diabetic.
3. **Recall**: Out of all the genuinely diabetic patients, how many did the model correctly identify. This is extremely important in healthcare models to ensure fewer false negatives.
4. **Confusion Matrix**: A visual matrix detailing True Positives, True Negatives, False Positives, and False Negatives.

### Step 4: Storage & Architecture (`src/database.py`)
To make this a full-featured application, we utilized **SQLite3** to store patient inputs and prediction outputs. A table named `patients` holds a record of every prediction made, creating an internal history/audit log.

### Step 5: User Interface Integration (`app.py`)
The system employs **Streamlit** to bridge the Python machine learning backend with a clean user-accessible web interface. Users can input new patient data, test different models, and instantly query the SQLite database.

## 3. Future Improvements
- **Model Expansion**: Integrating models like XGBoost or Support Vector Machines (SVM).
- **Deep Learning**: Implementing a neural network architecture using TensorFlow/Keras for complex non-linear feature interactions.
- **Cloud Database**: Migrating SQLite to PostgreSQL/MySQL and deploying the app via Docker on platforms like AWS or Heroku.
- **Dynamic Dataset Update**: Allowing doctors/admins to upload new batch CSV datasets through the UI to retrain the model dynamically.
