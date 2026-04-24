# ============================================================
# CROP RECOMMENDATION SYSTEM — STREAMLIT APP
# ============================================================
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    
    /* Title styling */
    .title-text {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        text-align: center;
        color: #b0bec5;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(86, 171, 47, 0.3);
    }
    
    .result-card h2 {
        color: #fff;
        font-size: 1.8rem;
        margin: 0;
    }
    
    .result-card p {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Info card */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Input sections */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #a8e063;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #56ab2f;
        padding-bottom: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #56ab2f, #a8e063) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.6) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #78909c;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Load Model and Encoder ---
@st.cache_resource
def load_model():
    """Load the trained model and label encoder."""
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


# Crop emoji mapping for visual appeal
CROP_EMOJIS = {
    "rice": "🍚", "wheat": "🌾", "maize": "🌽", "chickpea": "🫘",
    "kidneybeans": "🫘", "pigeonpeas": "🫛", "mothbeans": "🫘",
    "mungbean": "🫘", "blackgram": "🫘", "lentil": "🫘",
    "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "apple": "🍎", "orange": "🍊", "papaya": "🍈",
    "coconut": "🥥", "cotton": "🧶", "jute": "🌿", "coffee": "☕",
}


# --- App Header ---
st.markdown('<p class="title-text">🌾 Crop Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Enter soil and climate conditions to get the best crop recommendation</p>', unsafe_allow_html=True)

# --- Load the model ---
try:
    model, encoder = load_model()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run the Jupyter Notebook first to train and save the model.")
    st.info("Required files: `crop_model.pkl` and `label_encoder.pkl`")
    st.stop()

# --- Input Section ---
st.markdown('<p class="section-header">🧪 Soil Nutrients (N-P-K)</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0, max_value=200, value=50, step=1,
        help="Nitrogen content ratio in soil"
    )

with col2:
    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0, max_value=200, value=50, step=1,
        help="Phosphorus content ratio in soil"
    )

with col3:
    potassium = st.number_input(
        "Potassium (K)",
        min_value=0, max_value=300, value=50, step=1,
        help="Potassium content ratio in soil"
    )

st.markdown("---")
st.markdown('<p class="section-header">🌤️ Climate Conditions</p>', unsafe_allow_html=True)
col4, col5 = st.columns(2)

with col4:
    temperature = st.slider(
        "🌡️ Temperature (°C)",
        min_value=0.0, max_value=50.0, value=25.0, step=0.1,
        help="Average temperature in degree Celsius"
    )

with col5:
    humidity = st.slider(
        "💧 Humidity (%)",
        min_value=0.0, max_value=100.0, value=70.0, step=0.1,
        help="Relative humidity in percentage"
    )

col6, col7 = st.columns(2)

with col6:
    ph = st.slider(
        "⚗️ Soil pH",
        min_value=0.0, max_value=14.0, value=6.5, step=0.1,
        help="pH value of the soil (0-14)"
    )

with col7:
    rainfall = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0, max_value=500.0, value=150.0, step=1.0,
        help="Average rainfall in millimeters"
    )

# --- Prediction Button ---
st.markdown("---")

if st.button("🔍 Recommend Crop", use_container_width=True):
    # Prepare input array
    input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

    # Make prediction
    prediction = model.predict(input_features)
    crop_name = encoder.inverse_transform(prediction)[0]
    emoji = CROP_EMOJIS.get(crop_name, "🌱")

    # Show result with styled card
    st.markdown(f"""
    <div class="result-card">
        <h2>{emoji} {crop_name.upper()} {emoji}</h2>
        <p>Based on the given soil and climate conditions, <strong>{crop_name.title()}</strong> is the recommended crop.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show a summary of inputs
    st.markdown("### 📋 Input Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(f"""
        | Soil Nutrient | Value |
        |:---|:---|
        | Nitrogen (N) | {nitrogen} |
        | Phosphorus (P) | {phosphorus} |
        | Potassium (K) | {potassium} |
        """)

    with summary_col2:
        st.markdown(f"""
        | Climate Factor | Value |
        |:---|:---|
        | Temperature | {temperature}°C |
        | Humidity | {humidity}% |
        | Soil pH | {ph} |
        | Rainfall | {rainfall} mm |
        """)

    # Show prediction confidence if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_features)[0]
        top_3_idx = np.argsort(proba)[-3:][::-1]
        top_3_crops = encoder.inverse_transform(top_3_idx)
        top_3_proba = proba[top_3_idx]

        st.markdown("### 📊 Top 3 Predictions")
        for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_proba)):
            emoji_i = CROP_EMOJIS.get(crop, "🌱")
            medal = ["🥇", "🥈", "🥉"][i]
            st.progress(float(prob), text=f"{medal} {emoji_i} {crop.title()} — {prob * 100:.1f}%")

    st.success("✅ Prediction complete!")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🌾 Crop Recommendation System | Built with Streamlit & Scikit-learn</p>
    <p>Model: Random Forest Classifier | Dataset: Crop Recommendation Dataset</p>
</div>
""", unsafe_allow_html=True)
