import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Concrete Multi-Property Predictor", page_icon="🏗️", layout="wide")

st.title("🏗️ Concrete Multi-Property Predictor")
st.write("""
Predict key **Fresh** and **Hardened State** concrete properties using Machine Learning. 
Outputs include laboratory baseline estimates alongside **Field/Site-Calibrated** values.
""")

# --- Sidebar About Section ---
with st.sidebar:
    st.header("ℹ️ About the App")
    st.markdown("""
    **Concrete Multi-Property Predictor**  
    An AI-powered decision support tool bridging concrete materials science with computational machine learning.
    
    * **Algorithm:** Random Forest Regressor ($R^2 \\approx 0.88$, $\\text{MAE} \\approx 3.73\\text{ MPa}$)
    * **Key Outputs:** 
      - Compressive Strength ($f_{ck}$)
      - Flexural Modulus of Rupture ($f_{cr}$ via IS 456)
      - Fresh Slump (mm)
    * **Interpretability:** Explainable AI via SHAP Waterfall plots
    
    **Developed by:** Final Year B.E. Civil Engineering Project Batch (2027)
    """)
    st.divider()
    st.caption("Designed for rapid mix-design evaluation and quality control.")

# --- Model Training & Caching ---
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('concrete_data.csv')
        
        X = df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']]
        y = df['Strength']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, scaler, mae, r2

    except FileNotFoundError:
        st.error("Dataset 'concrete_data.csv' not found. Please place it in your project folder.")
        return None, None, None, None

model, scaler, mae, r2 = train_model()

# --- Application Layout ---
if model is not None:
    st.subheader("📊 Compressive Strength Model Accuracy (Test Data)")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} MPa")
    col_m2.metric("R² Score", f"{r2:.2f}")
    st.divider()

    # --- User Input Number Boxes ---
    st.subheader("🧪 Input Mix Design Proportions (Type exact values)")
    
    col_left, col_right = st.columns(2)

    with col_left:
        cement = st.number_input("Cement (kg/m³)", min_value=50.0, max_value=700.0, value=380.0, step=1.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=300.0, value=0.0, step=1.0)
        water = st.number_input("Water (kg/m³)", min_value=50.0, max_value=300.0, value=160.0, step=1.0)

    with col_right:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=500.0, max_value=1500.0, value=1100.0, step=1.0)
        fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=300.0, max_value=1200.0, value=700.0, step=1.0)
        age = st.number_input("Age (Days)", min_value=1, max_value=365, value=28, step=1)

    # --- Prediction & Calculations ---
    if st.button("Predict All Properties", type="primary"):
        feature_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                         'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
        
        input_df = pd.DataFrame([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]], 
                                columns=feature_names)
        
        # 1. Base Laboratory Prediction
        input_scaled = scaler.transform(input_df)
        lab_comp_strength = model.predict(input_scaled)[0]

        # 2. Site Calibration: Empirical Reduction Factor (k = 0.88)
        k_factor = 0.88
        site_comp_strength = lab_comp_strength * k_factor

        # 3. Flexural Strength (IS 456 formula on calibrated site strength: f_cr = 0.7 * sqrt(f_ck))
        flex_strength = 0.7 * np.sqrt(max(0, site_comp_strength))

        # 4. Fresh Concrete Slump Estimation (mm)
        wc_ratio = water / cement if cement > 0 else 0.5
        base_slump = (wc_ratio - 0.25) * 200 + (superplasticizer * 8.0)
        slump_mm = max(0.0, min(250.0, base_slump))

        # --- Display Results ---
        st.divider()
        st.subheader("🎯 Predicted Concrete Properties")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Lab Baseline Strength", f"{lab_comp_strength:.2f} MPa")
        col_res2.metric("Site Calibrated (f_ck)", f"{site_comp_strength:.2f} MPa", delta=f"-{(1-k_factor)*100:.0f}% Field Loss")
        col_res3.metric("Flexural Strength (f_cr)", f"{flex_strength:.2f} MPa")
        col_res4.metric("Workability (Slump)", f"{slump_mm:.0f} mm")

        # --- SHAP Plot ---
        st.divider()
        st.subheader("🧠 SHAP Feature Contribution (Compressive Strength)")
        st.write("Waterfall chart explaining parameter impact on the baseline laboratory model:")
        
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)
        shap_values.feature_names = feature_names
        shap_values.data = input_df.values
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # --- Export Report ---
        st.divider()
        st.subheader("📥 Export Mix Design & Prediction Report")
        
        report_data = {
            "Cement (kg/m3)": [cement],
            "Blast Furnace Slag (kg/m3)": [slag],
            "Fly Ash (kg/m3)": [fly_ash],
            "Water (kg/m3)": [water],
            "Superplasticizer (kg/m3)": [superplasticizer],
            "Coarse Aggregate (kg/m3)": [coarse_agg],
            "Fine Aggregate (kg/m3)": [fine_agg],
            "Age (Days)": [age],
            "Lab Baseline Strength (MPa)": [round(lab_comp_strength, 2)],
            "Site Calibrated Strength (MPa)": [round(site_comp_strength, 2)],
            "Predicted Flexural Strength (MPa)": [round(flex_strength, 2)],
            "Predicted Slump (mm)": [round(slump_mm, 1)]
        }
        
        report_df = pd.DataFrame(report_data)
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Complete Report (CSV)",
            data=csv_data,
            file_name="calibrated_concrete_report.csv",
            mime="text/csv",
            type="primary"
        )
