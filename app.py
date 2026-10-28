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
Enter your exact mix design proportions below and click predict.
""")

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

    # --- User Input Number Boxes (Replaced Sliders) ---
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
        
        input_scaled = scaler.transform(input_df)
        comp_strength = model.predict(input_scaled)[0]

        # Flexural Strength calculation (IS 456 formula)
        flex_strength = 0.7 * np.sqrt(max(0, comp_strength))

        # Slump estimation formula
        wc_ratio = water / cement if cement > 0 else 0.5
        base_slump = (wc_ratio - 0.25) * 200 + (superplasticizer * 8.0)
        slump_mm = max(0.0, min(250.0, base_slump))

        # --- Display Results ---
        st.divider()
        st.subheader("🎯 Predicted Concrete Properties")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Compressive Strength (f_ck)", f"{comp_strength:.2f} MPa")
        col_res2.metric("Flexural Strength (f_cr)", f"{flex_strength:.2f} MPa")
        col_res3.metric("Workability (Slump)", f"{slump_mm:.0f} mm")

        # --- SHAP Plot ---
        st.divider()
        st.subheader("🧠 SHAP Feature Contribution (Compressive Strength)")
        
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
            "Predicted Compressive Strength (MPa)": [round(comp_strength, 2)],
            "Predicted Flexural Strength (MPa)": [round(flex_strength, 2)],
            "Predicted Slump (mm)": [round(slump_mm, 1)]
        }
        
        report_df = pd.DataFrame(report_data)
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Complete Report (CSV)",
            data=csv_data,
            file_name="concrete_multi_property_report.csv",
            mime="text/csv",
            type="primary"
        )