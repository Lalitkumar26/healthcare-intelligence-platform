# app.py - Health Care Intelligence Platform (Pink, Sky Blue, White Theme)
# Includes enlarged hamburger menu icon for better visibility
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Health Care Intelligence", page_icon="🏥", layout="wide")

# ==================== SESSION STATE ====================
if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header – sky blue (from your code) */
    .custom-header {
        background-color: #00BFFF;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .custom-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 600;
    }
    .custom-header p {
        color: #fff0f5;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar – solid sky blue */
    [data-testid="stSidebar"] {
        background-color: #00BFFF;
        border-right: 1px solid #d0d7de;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b;
    }
    
    /* 🔽 ENLARGED HAMBURGER MENU ICON (sidebar toggle) 🔽 */
    [data-testid="stSidebarCollapsedControl"] {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #ff69b4;   /* pink background – makes it very noticeable */
        border-radius: 50%;
        margin: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: 0.2s;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        transform: scale(1.05);
        background-color: #ff1493;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        width: 28px;
        height: 28px;
        stroke: white;
        stroke-width: 2;
    }
    /* 🔼 END OF ENLARGED HAMBURGER ICON 🔼 */
    
    /* Cards */
    .card {
        background-color: white;
        border-left: 5px solid #87ceeb;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: 0.2s;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .card h3 {
        color: #ff69b4;
        margin-top: 0;
        font-weight: 600;
    }
    
    /* Buttons (prediction pages) */
    .stButton > button {
        background-color: #ff69b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #ff1493;
        transform: translateY(-1px);
    }
    
    /* Risk result cards */
    .risk-high {
        background-color: #ffe4e1;
        border-left: 5px solid #ff69b4;
        padding: 1rem;
        border-radius: 10px;
        color: #b71c1c;
        font-weight: bold;
        text-align: center;
    }
    .risk-low {
        background-color: #e0f7fa;
        border-left: 5px solid #87ceeb;
        padding: 1rem;
        border-radius: 10px;
        color: #00695c;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #fff0f5;
        border-left: 4px solid #ff69b4;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Home page cards */
    .clickable-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: 0.2s;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    .clickable-card:hover {
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        border-color: #ff69b4;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.2rem;
        margin-top: 2rem;
        background-color: #f8f9fa;
        border-top: 1px solid #e0e0e0;
        font-size: 0.85rem;
        color: #555;
        border-radius: 8px;
    }
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    models = {}
    scalers = {}
    try:
        models["dia"] = joblib.load("models/diabetes_model.pkl")
        scalers["dia"] = joblib.load("models/diabetes_scaler.pkl")
    except:
        pass
    try:
        models["heart"] = joblib.load("models/heart_model.pkl")
        scalers["heart"] = joblib.load("models/heart_scaler.pkl")
    except:
        pass
    try:
        models["read"] = joblib.load("models/readmission_model.pkl")
        scalers["read"] = joblib.load("models/readmission_scaler.pkl")
    except:
        pass
    try:
        models["clus"] = joblib.load("models/clustering_model.pkl")
        scalers["clus"] = joblib.load("models/clustering_scaler.pkl")
    except:
        pass
    return models, scalers

models, scalers = load_models()

# ==================== HEADER ====================
st.markdown("""
<div class="custom-header">
    <h1>🏥 Health Care Intelligence Platform</h1>
    <p>AI-Powered Medical Predictions & Analytics</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.markdown("### 🧠 Navigation")
    menu = st.selectbox(
        "Select Module",
        ["🏠 Home", "🩺 Diabetes Prediction", "❤️ Heart Disease Prediction",
         "🏥 Readmission Prediction", "🔍 Disease Clustering",
         "💊 Medicine Recommendation", "🧮 BMI Calculator"],
        index=["🏠 Home", "🩺 Diabetes Prediction", "❤️ Heart Disease Prediction",
               "🏥 Readmission Prediction", "🔍 Disease Clustering",
               "💊 Medicine Recommendation", "🧮 BMI Calculator"].index(st.session_state.menu)
    )
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("This platform uses Machine Learning to provide accurate health predictions. All models are trained on real medical datasets.")
    st.markdown("### 👨‍⚕️ Disclaimer")
    st.warning("For educational purposes only. Always consult a doctor.")
    
    if menu != st.session_state.menu:
        st.session_state.menu = menu
        st.rerun()

# ==================== HOME PAGE (6 Clickable Cards with Emoji Images) ====================
if st.session_state.menu == "🏠 Home":
    st.markdown("### Welcome to the Health Care Intelligence Platform")
    st.markdown("Select a module from the sidebar or click any card below to get started.")
    
    # Row 1: Diabetes, Heart, Readmission
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🩺</h1>", unsafe_allow_html=True)
        if st.button("**Diabetes Prediction**\n\nPredict risk using 8 clinical measurements", use_container_width=True):
            st.session_state.menu = "🩺 Diabetes Prediction"
            st.rerun()
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>❤️</h1>", unsafe_allow_html=True)
        if st.button("**Heart Disease**\n\nAssess cardiovascular risk", use_container_width=True):
            st.session_state.menu = "❤️ Heart Disease Prediction"
            st.rerun()
    with col3:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🏥</h1>", unsafe_allow_html=True)
        if st.button("**Readmission**\n\n30‑day hospital readmission risk", use_container_width=True):
            st.session_state.menu = "🏥 Readmission Prediction"
            st.rerun()
    
    # Row 2: Clustering, Medicine, BMI
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🔍</h1>", unsafe_allow_html=True)
        if st.button("**Disease Clustering**\n\nGroup patients by health patterns", use_container_width=True):
            st.session_state.menu = "🔍 Disease Clustering"
            st.rerun()
    with col5:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>💊</h1>", unsafe_allow_html=True)
        if st.button("**Medicine Recommendation**\n\nGet drug suggestions by condition", use_container_width=True):
            st.session_state.menu = "💊 Medicine Recommendation"
            st.rerun()
    with col6:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🧮</h1>", unsafe_allow_html=True)
        if st.button("**BMI Calculator**\n\nCheck your Body Mass Index", use_container_width=True):
            st.session_state.menu = "🧮 BMI Calculator"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    st.markdown("""
    - ✅ **7 AI modules** for different health predictions
    - ✅ **Real-time predictions** – just enter patient data
    - ✅ **Evidence-based recommendations**
    - ✅ **BMI Calculator** included
    - ✅ **Mobile responsive** design
    """)

# ==================== DIABETES PREDICTION ====================
elif st.session_state.menu == "🩺 Diabetes Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Diabetes Risk Assessment")
    st.markdown("Enter the patient's health metrics below:")
    
    if "dia" not in models:
        st.error("❌ Diabetes model not available. Please train first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose", 0, 300, 100)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)
        
        if st.button("🔍 Analyze Diabetes Risk", use_container_width=True):
            inp = scalers["dia"].transform([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            pred = models["dia"].predict(inp)[0]
            prob = models["dia"].predict_proba(inp)[0]
            if pred == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH RISK of Diabetes – Probability: {prob[1]:.2%}</div>', unsafe_allow_html=True)
                st.warning("Recommendation: Consult a doctor for further evaluation.")
            else:
                st.markdown(f'<div class="risk-low">✅ LOW RISK of Diabetes – Probability: {prob[0]:.2%}</div>', unsafe_allow_html=True)
                st.success("Recommendation: Maintain healthy lifestyle.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== HEART DISEASE ====================
elif st.session_state.menu == "❤️ Heart Disease Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("❤️ Heart Disease Risk Assessment")
    if "heart" not in models:
        st.error("❌ Heart model not available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", ["Female", "Male"])
            sex_val = 0 if sex == "Female" else 1
            cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
            cp_val = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp)
            trestbps = st.number_input("Resting BP (mm Hg)", 50, 250, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
            restecg_val = ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"].index(restecg)
        with col2:
            thalach = st.number_input("Max Heart Rate", 50, 250, 150)
            exang = st.selectbox("Exercise Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1)
            slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
            slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
            ca = st.number_input("Major Vessels (0-4)", 0, 4, 0)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect", "Not described"])
            thal_val = ["Normal", "Fixed defect", "Reversible defect", "Not described"].index(thal)
        if st.button("🔍 Analyze Heart Risk", use_container_width=True):
            inp = scalers["heart"].transform([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            pred = models["heart"].predict(inp)[0]
            prob = models["heart"].predict_proba(inp)[0]
            if pred == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH RISK – Probability: {prob[1]:.2%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ LOW RISK – Probability: {prob[0]:.2%}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== READMISSION ====================
elif st.session_state.menu == "🏥 Readmission Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏥 Hospital Readmission Risk (<30 days)")
    if "read" not in models:
        st.error("❌ Readmission model not available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            time_hosp = st.number_input("Time in hospital (days)", 1, 30, 5)
            lab = st.number_input("Lab procedures", 0, 200, 50)
            proc = st.number_input("Procedures", 0, 20, 2)
            meds = st.number_input("Medications", 1, 100, 15)
        with col2:
            out = st.number_input("Outpatient visits (prior year)", 0, 50, 1)
            inp_vis = st.number_input("Inpatient visits (prior year)", 0, 20, 1)
            emerg = st.number_input("Emergency visits (prior year)", 0, 20, 0)
            age = st.number_input("Age", 18, 100, 65)
        if st.button("🔍 Analyze Readmission Risk", use_container_width=True):
            inp = scalers["read"].transform([[time_hosp, lab, proc, meds, out, inp_vis, emerg, age]])
            pred = models["read"].predict(inp)[0]
            prob = models["read"].predict_proba(inp)[0]
            if pred == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH READMISSION RISK – Probability: {prob[1]:.2%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ LOW READMISSION RISK – Probability: {prob[0]:.2%}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CLUSTERING ====================
elif st.session_state.menu == "🔍 Disease Clustering":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Patient Disease Clustering")
    if "clus" not in models:
        st.error("❌ Clustering model not available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 0)
            glu = st.number_input("Glucose", 0, 300, 100)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
        with col2:
            ins = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)
        if st.button("🔍 Find Cluster", use_container_width=True):
            inp = scalers["clus"].transform([[preg, glu, bp, skin, ins, bmi, dpf, age]])
            cluster = models["clus"].predict(inp)[0]
            if cluster == 0:
                st.markdown('<div class="risk-low">🟢 CLUSTER 0: Low risk – Healthy</div>', unsafe_allow_html=True)
            elif cluster == 1:
                st.markdown('<div class="info-box">🟡 CLUSTER 1: Moderate risk – Monitor regularly</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-high">🔴 CLUSTER 2: High risk – Needs immediate attention</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== MEDICINE RECOMMENDATION ====================
elif st.session_state.menu == "💊 Medicine Recommendation":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💊 Medicine Recommendation System")
    med_db = {
        "Diabetes": ["Metformin 500mg – twice daily", "Insulin (as prescribed)", "Glipizide 5mg – before meals"],
        "Hypertension": ["Lisinopril 10mg – once daily", "Amlodipine 5mg – once daily", "Metoprolol 50mg – twice daily"],
        "Heart Disease": ["Aspirin 81mg – once daily", "Atorvastatin 20mg – once daily", "Metoprolol 25mg – twice daily"],
        "High Cholesterol": ["Simvastatin 20mg – bedtime", "Atorvastatin 40mg – once daily", "Fenofibrate 145mg – once daily"],
        "Fever": ["Acetaminophen 500mg – every 4-6h", "Ibuprofen 400mg – every 6-8h", "Rest and fluids"],
        "Headache": ["Acetaminophen 500mg – every 4-6h", "Ibuprofen 200mg – every 6-8h", "Cold compress"],
        "Cough": ["Dextromethorphan 15mg – every 4h", "Honey and warm water", "Stay hydrated"],
        "Cold": ["Rest and fluids", "Pseudoephedrine 30mg – for congestion", "Throat lozenges"]
    }
    disease = st.selectbox("Select Disease/Condition", list(med_db.keys()))
    if st.button("💊 Get Recommendations", use_container_width=True):
        st.subheader(f"Recommended medicines for {disease}:")
        for i, med in enumerate(med_db[disease], 1):
            st.write(f"{i}. {med}")
        st.warning("⚠️ Always consult a doctor before taking any medication.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== BMI CALCULATOR ====================
elif st.session_state.menu == "🧮 BMI Calculator":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧮 BMI Calculator")
    st.markdown("Calculate Body Mass Index (BMI) and assess weight status.")
    
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
        height_m = height_cm / 100
    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, step=1)
    
    if st.button("Calculate BMI", use_container_width=True):
        if height_m > 0:
            bmi = weight_kg / (height_m ** 2)
            st.metric("Your BMI", f"{bmi:.1f}")
            
            if bmi < 18.5:
                category = "Underweight"
                risk = "Nutritional deficiency risk – consider balanced diet"
                color = "info-box"
            elif 18.5 <= bmi < 25:
                category = "Normal weight"
                risk = "Low health risk – keep it up!"
                color = "risk-low"
            elif 25 <= bmi < 30:
                category = "Overweight"
                risk = "Moderate risk – weight management advised"
                color = "info-box"
            else:
                category = "Obese"
                risk = "High risk – consult a doctor"
                color = "risk-high"
            
            st.markdown(f'<div class="{color}"><strong>Category:</strong> {category}<br><strong>Health Risk:</strong> {risk}</div>', unsafe_allow_html=True)
            
            min_ideal = 18.5 * (height_m ** 2)
            max_ideal = 24.9 * (height_m ** 2)
            st.markdown(f"Ideal weight range for your height: **{min_ideal:.1f} kg – {max_ideal:.1f} kg**")
        else:
            st.error("Height must be > 0")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🏥 Health Care Intelligence Platform | Powered by Machine Learning</p>
    <p>© 2025 | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)