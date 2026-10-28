🏗️ Concrete Multi-Property Predictor
A data-driven civil engineering web application built using Python and Streamlit to predict the Compressive Strength, Flexural Strength, and Fresh Slump of concrete mixes
Powered by a non-linear Random Forest Regressor trained on standardized mix design data, the application eliminates standard 28-day laboratory testing delays by providing near-instantaneous property estimations, model interpretability via SHAP waterfall plots, and empirical site calibration.

📌 Key Features

Multi-Property Estimation: Predicts hardened Compressive Strength (f_ck), Flexural Strength (f_cr via IS 456), and fresh-state Slump (mm) simultaneously
Site Calibration (k=0.88k = 0.88): Calibrates idealized laboratory baseline predictions down to realistic construction-site conditions, accounting for ~12% field losses from placement, compaction, and curing variances.
Explainable AI (SHAP): Features interactive SHAP waterfall plots to eliminate "black-box" predictions, showing the positive or negative contribution of each constituent.
Exportable Quality Reports: Includes one-click CSV report generation of input mix proportions and calculated strength values for engineering documentation.


🧪 Input Parameters (8 Constituents)

Cement (kg/m^3)
Blast Furnace Slag (kg/m^3)
Fly Ash (kg/m^3)
Water (kg/m^3)
Superplasticizer (kg/m^3)
Coarse Aggregate (kg/m^3)
Fine Aggregate (kg/m^3)
Age (Days, 1 to 365)


⚙️ Tech Stack & Dependencies

Language: Python
Frontend UI: Streamlit
Machine Learning: Scikit-Learn
Interpretability: SHAP (TreeExplainer)
Data Processing & Visualization: Pandas, NumPy, Matplotlib


🚀 How to Run Locally

Clone or download this repository to your local computer.
Install the required Python packages:
pip install -r requirements.txt
