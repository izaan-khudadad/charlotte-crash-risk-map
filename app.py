import streamlit as st
import pandas as pd
import joblib
import base64

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load image as background
def get_base64_of_image(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_of_image("charlotte_color.png")

# Inject CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: contain;
        background-repeat: repeat;
        background-position: top left;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="text-align:center;">
    <div style="
        padding: 15px 25px;
        background: rgba(14, 17, 23, 0.75);
        border-radius: 12px;
        display: inline-block;
        backdrop-filter: blur(6px);
    ">
        <h1 style="
            color: #F5F7FA;
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
        ">
         Charlotte Crash Severity Risk Map
        </h1>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("xgb_model.pkl")

# Load data
df = pd.read_csv("cleaned_data.csv")

# Sidebar controls
st.sidebar.header("Simulation Controls")

# Helper data
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def format_hour(h):
    suffix = "AM" if h < 12 else "PM"
    h_display = h % 12
    if h_display == 0:
        h_display = 12
    return f"{h_display} {suffix}"

# Sliders (same functionality)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.slider("Day of Week", 0, 6, 3)
visibility = st.sidebar.slider("Visibility (miles)", 0.1, 10.0, 5.0)

# Clean labels (this is what replaces the ugly numbers)
st.sidebar.markdown("### Selected Conditions")
st.sidebar.markdown(f"🕒 **Time:** {format_hour(hour)}")
st.sidebar.markdown(f"📅 **Day:** {days[day]}")
st.sidebar.markdown(f"👁️ **Visibility:** {visibility:.2f} miles")



import numpy as np

# Create a copy
df_sim = df.copy()

# Update time features
df_sim["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df_sim["hour_cos"] = np.cos(2 * np.pi * hour / 24)

df_sim["day_sin"] = np.sin(2 * np.pi * day / 7)
df_sim["day_cos"] = np.cos(2 * np.pi * day / 7)

# Update visibility
df_sim["hourly_visibility"] = visibility

# Prepare features
X_sim = df_sim.drop(columns=["crash_severity", "severe_crash"])

# Predict risk
with st.spinner("🚗 Calculating traffic risk across Charlotte..."):
    df_sim["risk_score"] = model.predict_proba(X_sim)[:,1]

# Aggregate by location
map_data = df_sim.groupby(["latitude", "longitude"])["risk_score"].mean().reset_index()

import plotly.express as px

fig = px.scatter_mapbox(
    map_data.sample(20000),  # 👈 reduces clutter & speeds up app
    lat="latitude",
    lon="longitude",
    color="risk_score",
    color_continuous_scale="Reds",
    zoom=10,
    height=600
)

fig.update_layout(
    mapbox_style="carto-positron",  # cleaner background
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.markdown("""
<div style="
    border: 0px solid rgba(77, 163, 255, 0.4);
    border-radius: 0px;
    background: rgba(14, 17, 23, 0.6);
    box-shadow: 0 0 0px rgba(77, 163, 255, 0.15);
">
""", unsafe_allow_html=True)

st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)