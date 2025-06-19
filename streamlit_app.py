import streamlit as st
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="🌾 Crop Recommender", layout="centered")
st.title("🌾 Weather-Based Crop Recommendation System")

# Paths to model and encoder
MODEL_PATH = "artifacts/crop_recommendation_model.pkl"
ENCODER_PATH = "artifacts/label_encoder.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    return None, None

# Load model and encoder
model, encoder = load_model()

if not model or not encoder:
    st.error("❌ Model or encoder not found. Please train the model and try again.")
else:
    st.subheader("🌦️ Enter Weather and Soil Conditions")

    # User inputs
    N = st.number_input("Nitrogen (N)", 0, 140, 90)
    P = st.number_input("Phosphorus (P)", 5, 145, 42)
    K = st.number_input("Potassium (K)", 5, 205, 43)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.number_input("Soil pH", 3.0, 10.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

    if st.button("🌱 Recommend Crop"):
        try:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            crop = encoder.inverse_transform(prediction)[0]
            st.success(f"✅ Recommended Crop: **{crop.upper()}**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
