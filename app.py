import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import random
from sklearn.neural_network import MLPRegressor

# Load ML models and dataset
rf_model = joblib.load("catalyst_model.pkl")

data = pd.read_csv(r"data\nanoparticle_catalytic_dataset_5000.csv")

# Harmonize dataset column names
if "Crystallite_Size_nm" not in data.columns and "size_nm" in data.columns:
    data["Crystallite_Size_nm"] = data["size_nm"]

if "Surface_Area" not in data.columns and "surface_area" in data.columns:
    data["Surface_Area"] = data["surface_area"]

if "Catalytic_Activity" not in data.columns and "catalytic_activity" in data.columns:
    data["Catalytic_Activity"] = data["catalytic_activity"] * 100.0

for col_name, default in {
    "Peak_2theta": 35.0,
    "FWHM": 0.35,
    "Lattice_Param": 5.0,
    "Intensity": 450.0,
}.items():
    if col_name not in data.columns:
        data[col_name] = default

feature_names = [
    "Peak_2theta",
    "FWHM",
    "Crystallite_Size_nm",
    "Lattice_Param",
    "Intensity",
    "Surface_Area",
]

X = data[feature_names]
y = data["Catalytic_Activity"]

# Deep learning model
dl_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
)

dl_model.fit(X, y)

st.set_page_config(
    page_title="AI Nanocatalyst Research Lab",
    layout="wide"
)

# LIGHT THEME
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
h1 { color:#0f172a; font-weight:700; }
h2,h3 { color:#1e293b; }
[data-testid="stMetricValue"] { color:#2563eb; font-size:32px; }
.block-container { padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🔬 AI Nanocatalyst Research Laboratory")

st.write("""
AI powered platform for **predicting, discovering and optimizing nanoparticle catalysts**
using machine learning and structural parameters derived from XRD.
""")

st.markdown("---")

# IMAGE
image_path = "assets/nanocat.png"

if os.path.exists(image_path):
    st.image(image_path, width=300)
else:
    st.warning("Nanocatalyst image not found.")

# SIDEBAR
st.sidebar.header("⚙ Catalyst Structural Parameters")

nanoparticle = st.sidebar.selectbox(
    "Nanoparticle",
    ["CuO", "ZnO", "TiO₂", "Ag", "Au"],
)

Peak_2theta = st.sidebar.slider("Peak 2θ", 20.0, 80.0, 35.0)
FWHM = st.sidebar.slider("FWHM", 0.1, 1.0, 0.35)
Crystallite_Size_nm = st.sidebar.slider("Crystallite Size (nm)", 5.0, 100.0, 25.0)
Lattice_Param = st.sidebar.slider("Lattice Parameter", 3.0, 9.0, 5.0)
Intensity = st.sidebar.slider("Peak Intensity", 100, 1000, 450)
Surface_Area = st.sidebar.slider("Surface Area", 10, 200, 70)

features = np.array([[
    Peak_2theta,
    FWHM,
    Crystallite_Size_nm,
    Lattice_Param,
    Intensity,
    Surface_Area,
]])

rf_prediction = rf_model.predict(features)[0]
dl_prediction = dl_model.predict(features)[0]

# METRICS
col1, col2, col3 = st.columns(3)

col1.metric("RandomForest Prediction", f"{rf_prediction:.2f}%")
col2.metric("Deep Learning Prediction", f"{dl_prediction:.2f}%")
col3.metric("Surface Area", f"{Surface_Area} m²/g")

st.markdown("---")

# AI RECOMMENDATION
st.subheader("🤖 AI Catalyst Recommendation")

score = (rf_prediction + dl_prediction) / 2

if score > 85:
    st.success("Highly efficient catalyst predicted.")
elif score > 70:
    st.info("Moderate catalytic performance.")
else:
    st.warning("Optimization recommended.")

st.markdown("---")

# XRD SIMULATION
st.subheader("📈 Simulated XRD Pattern")

x = np.linspace(20,80,300)
y = np.exp(-(x-Peak_2theta)**2/5)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_xlabel("2θ")
ax.set_ylabel("Intensity")

st.pyplot(fig)

st.markdown("---")

# FEATURE IMPORTANCE
st.subheader("📊 Feature Importance")

importances = rf_model.feature_importances_

fig2, ax2 = plt.subplots()
ax2.barh(feature_names, importances)

st.pyplot(fig2)

st.markdown("---")

# HEATMAP
st.subheader("📊 Catalytic Activity Heatmap")

surface_range = np.linspace(10, 200, 25)
size_range = np.linspace(5, 100, 25)

heat = []

for s in size_range:
    row = []
    for a in surface_range:

        test = np.array([[

            Peak_2theta,
            FWHM,
            s,
            Lattice_Param,
            Intensity,
            a,

        ]])

        row.append(rf_model.predict(test)[0])

    heat.append(row)

heat_df = pd.DataFrame(heat, index=size_range, columns=surface_range)

fig_heat = px.imshow(
    heat_df,
    labels=dict(x="Surface Area", y="Crystallite Size", color="Activity"),
)

st.plotly_chart(fig_heat)

st.markdown("---")

# AI OPTIMIZATION
st.subheader("🔬 AI Catalyst Optimization")

best_score = -1e9
best = None

for _ in range(300):

    test = np.array([[

        random.uniform(20, 80),
        random.uniform(0.1, 1),
        random.uniform(5, 100),
        random.uniform(3, 9),
        random.uniform(100, 1000),
        random.uniform(10, 200),

    ]])

    score = rf_model.predict(test)[0]

    if score > best_score:
        best_score = score
        best = test

st.write("Best Predicted Catalyst")
st.write("Predicted Activity:", round(best_score, 2))
st.dataframe(pd.DataFrame(best, columns=feature_names))

st.markdown("---")

# AI INVERSE DESIGN
st.subheader("🎯 AI Inverse Catalyst Design")

target = st.slider("Target Catalytic Activity", 50, 100, 85)

best_diff = float("inf")
best_design = None

for _ in range(500):

    candidate = np.array([[

        random.uniform(20, 80),
        random.uniform(0.1, 1),
        random.uniform(5, 100),
        random.uniform(3, 9),
        random.uniform(100, 1000),
        random.uniform(10, 200),

    ]])

    pred = rf_model.predict(candidate)[0]
    diff = abs(pred - target)

    if diff < best_diff:
        best_diff = diff
        best_design = candidate

st.write("Structure predicted to reach target activity")
st.dataframe(pd.DataFrame(best_design, columns=feature_names))

st.markdown("---")

# AUTOMATIC DISCOVERY
st.subheader("🧠 Automatic Catalyst Discovery")

discoveries = []

for _ in range(50):

    candidate = np.array([

        random.uniform(20, 80),
        random.uniform(0.1, 1),
        random.uniform(5, 100),
        random.uniform(3, 9),
        random.uniform(100, 1000),
        random.uniform(10, 200),

    ])

    score = rf_model.predict(candidate.reshape(1, -1))[0]

    discoveries.append(list(candidate) + [score])

df_discover = pd.DataFrame(discoveries, columns=feature_names + ["Activity"])
df_discover = df_discover.sort_values("Activity", ascending=False)

st.dataframe(df_discover.head(10))

st.markdown("---")

# 3D LANDSCAPE
st.subheader("🌍 Catalytic Activity Landscape")

fig3 = px.scatter_3d(
    data.sample(min(500, len(data))),
    x="Crystallite_Size_nm",
    y="Surface_Area",
    z="Catalytic_Activity",
    color="Catalytic_Activity",
)

st.plotly_chart(fig3)

st.markdown("---")

# FILE UPLOAD
st.subheader("📂 Upload XRD Dataset")

uploaded = st.file_uploader("Upload CSV")

if uploaded:

    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    missing = [c for c in feature_names if c not in df.columns]

    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")

    else:
        df_model = df[feature_names]
        preds = rf_model.predict(df_model)

        df["Predicted Activity"] = preds

        st.dataframe(df)

        st.success("Prediction completed")