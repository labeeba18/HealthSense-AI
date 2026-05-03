import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from src.database import init_db, insert_patient, get_all_patients
import os
import time

# Set page configuration with a modern dark layout
st.set_page_config(
    page_title="HealthSense AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep dark mode CSS, Card styling, and hide Sidebar completely
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide the Streamlit Sidebar completely */
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Remove white gap at the top and avoid scrolling */
    header[data-testid="stHeader"] {display: none;}
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 95%;
    }

    /* Global background and font tweaks */
    .stApp {
        background-color: #0f172a;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Force all text and labels to be bright white so they are perfectly visible */
    p, label, h1, h2, h3, h4, h5, h6, .st-emotion-cache-1629p8f p {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }
    .card-text {
        color: #94a3b8 !important; 
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Modern Card Layouts */
    .feature-card {
        background-color: #1e293b;
        border-radius: 16px;
        padding: 24px 20px;
        margin: 10px 0px;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        animation: fadeIn 0.8s ease-in-out;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        border-color: #2563eb;
    }
    .card-title {
        color: #60a5fa !important;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Primary Button color override to fit theme */
    button[kind="primary"] {
        background-color: #2563eb !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4) !important;
    }
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Secondary Button override */
    button[kind="secondary"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: #334155 !important;
        border-color: #475569 !important;
        transform: translateY(-2px) !important;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: #22c55e !important; 
        font-size: 38px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] > div > div > p {
        color: #94a3b8 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Form centering container */
    .form-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Styled Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
        text-align: left;
        color: #f8fafc;
        background-color: #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .styled-table thead tr {
        background-color: #0f172a;
        color: #60a5fa;
        border-bottom: 2px solid #334155;
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #334155;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #0f172a;
    }
    .styled-table tbody tr:hover {
        background-color: #334155;
        transition: background-color 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Application session state for Step-by-Step navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

def go_to_predict():
    st.session_state.page = "predict"
    
def go_to_dashboard():
    st.session_state.page = "dashboard"

def go_home():
    st.session_state.page = "home"

# Initialize Database
init_db()

@st.cache_resource(show_spinner=False)
def load_models_and_scaler():
    scaler = joblib.load('models/scaler.pkl')
    # Default to the highly accurate Random Forest, removing confusing dropdowns
    model = joblib.load('models/Random_Forest_model.pkl')
    with open('models/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    return scaler, model, eval_results

try:
    scaler, model, eval_results = load_models_and_scaler()
except FileNotFoundError:
    st.error("Model files not found! Please run the training scripts first.")
    st.stop()


# ==========================================
# PAGE 1: HOME LANDING
# ==========================================
if st.session_state.page == "home":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 45px; margin-top: -10px;'>HealthSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e !important; margin-bottom: 25px; font-size: 18px;'>Enterprise-grade disease prediction and health analytics platform.</p>", unsafe_allow_html=True)
    
    # Next Step Navigation directly below text to avoid scrolling
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        st.button("Get Started Now", on_click=go_to_predict, type="primary", use_container_width=True)
    
    # Feature Cards layout
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-card"><div class="card-title">🔍 AI Detection</div><div class="card-text">Early detection with extreme accuracy for rapid intervention.</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><div class="card-title">📋 Personalized</div><div class="card-text">Correlate your metrics against global data immediately.</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><div class="card-title">📊 Analytics</div><div class="card-text">Monitor real-time charts and metric insights seamlessly.</div></div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2: DIABETES PREDICTION
# ==========================================
elif st.session_state.page == "predict":
    
    col_btn, empty = st.columns([1, 5])
    with col_btn:
        st.button("← Back", on_click=go_home)
    
    st.markdown("<h2 style='text-align:center; font-size: 30px; margin-top: -20px;'>🩺 Condition Prediction Form</h2>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        r1c1, r1c2 = st.columns(2)
        pregnancies = r1c1.number_input("Pregnancies", 0, 20, 0)
        glucose = r1c2.number_input("Glucose Level", 0.0, 300.0, 120.0)
        
        r2c1, r2c2 = st.columns(2)
        blood_pressure = r2c1.number_input("Blood Pressure", 0.0, 250.0, 70.0)
        skin_thickness = r2c2.number_input("Skin Thickness", 0.0, 150.0, 25.0)
        
        r3c1, r3c2 = st.columns(2)
        insulin = r3c1.number_input("Insulin Level", 0.0, 1000.0, 80.0)
        bmi = r3c2.number_input("BMI", 0.0, 100.0, 25.0)
        
        r4c1, r4c2 = st.columns(2)
        diabetes_pedigree = r4c1.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
        age = r4c2.number_input("Age", 1, 120, 35)
        
        # Button directly generates report
        submit_btn = st.form_submit_button("Analyze Patient Data", type="primary", use_container_width=True)
        
    if submit_btn:
        with st.spinner("Analyzing Patient Data..."):
            time.sleep(2)
            
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        scaled_features = scaler.transform(features)
        
        prediction_num = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        result_text = "Diabetic (High Risk)" if prediction_num == 1 else "Healthy (Low Risk)"
        bg_color = "#450a0a" if prediction_num == 1 else "#052e16"
        border_color = "#ef4444" if prediction_num == 1 else "#22c55e"
        conf = prediction_proba[1]*100 if prediction_num == 1 else prediction_proba[0]*100
        
        insert_patient(features[0], result_text)
        
        st.session_state.pred_result = {
            'prediction_num': prediction_num,
            'result_text': result_text,
            'bg_color': bg_color,
            'border_color': border_color,
            'conf': conf
        }
        st.session_state.page = "result"
        st.rerun()


# ==========================================
# PAGE 3: AI DIAGNOSIS REPORT
# ==========================================
elif st.session_state.page == "result":
    
    col_btn, empty = st.columns([1, 5])
    with col_btn:
        st.button("← Back to Form", on_click=go_to_predict)
    
    st.markdown("<h2 style='text-align:center; font-size: 30px; margin-top: -20px;'>📄 AI Diagnosis Report</h2>", unsafe_allow_html=True)
    
    if 'pred_result' in st.session_state:
        res = st.session_state.pred_result
        
        st.markdown(f"""
        <div style="background:{res['bg_color']}; border-left:5px solid {res['border_color']}; padding: 25px; border-radius: 8px; margin-top: 15px; text-align: center;">
            <h3 style='margin:0; font-size:32px;'>{'⚠️' if res['prediction_num'] == 1 else '✅'} {res['result_text']}</h3>
            <p style="margin:10px 0 0 0; color: white; font-size: 20px;">Confidence Level: <strong>{res['conf']:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.toast("Record securely saved to database!", icon="🔒")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.5, 2, 1.5])
        with col2:
            st.button("View Global Dashboard Analytics ➔", on_click=go_to_dashboard, type="primary", use_container_width=True)
    else:
        st.error("No result found. Please submit the form first.")
        st.button("Go to Predict", on_click=go_to_predict)


# ==========================================
# PAGE 4: DASHBOARD OUTPUT
# ==========================================
elif st.session_state.page == "dashboard":
    
    col_btn1, col_btn2, empty_d = st.columns([1, 1, 4])
    with col_btn1:
        st.button("← Back", on_click=go_to_predict)
    with col_btn2:    
        st.button("🏠 Home", on_click=go_home)
        
    st.markdown("<h2 style='margin-top:-10px;'>📊 AI Global Health Analytics</h2>", unsafe_allow_html=True)
    
    df_records = get_all_patients()
    
    if df_records.empty:
        st.warning("No records found in database. Make some predictions first!")
    else:
        # Top KPI Metrics using Streamlit columns
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(df_records)
        avg_age = round(df_records['age'].mean(), 1)
        positive_cases = len(df_records[df_records['prediction'].str.contains('Diabetic')])
        avg_glucose = round(df_records['glucose'].mean(), 1)
        
        col1.metric("Total Patients Analyzed", total_patients)
        col2.metric("Average Patient Age", avg_age)
        col3.metric("High-Risk Detections", positive_cases, delta_color="inverse")
        col4.metric("Avg Glucose Level", f"{avg_glucose} mg/dL")
        
        st.markdown("---")
        
        # Plotly Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<h4 style='color:white !important'>Patient Outcomes (Classes)</h4>", unsafe_allow_html=True)
            pie_fig = px.pie(df_records, names="prediction", hole=0.4, 
                             color_discrete_sequence=['#ef4444', '#22c55e'])
            pie_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(pie_fig, use_container_width=True)
            
        with chart_col2:
            st.markdown("<h4 style='color:white !important'>Age Distribution vs Outcome</h4>", unsafe_allow_html=True)
            hist_fig = px.histogram(df_records, x="age", color="prediction", nbins=15,
                                    barmode="group", color_discrete_sequence=['#ef4444', '#22c55e'])
            hist_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(hist_fig, use_container_width=True)
            
        st.markdown("<h4 style='color:white !important'>Detailed Patient Query Table</h4>", unsafe_allow_html=True)
        st.markdown(df_records.to_html(classes="styled-table", index=False), unsafe_allow_html=True)
