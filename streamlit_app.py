import streamlit as st
import numpy as np
import joblib
import os

# Set up page
st.set_page_config(page_title="Crop Recommender", layout="centered")
st.title("🌾 Weather-Based Crop Recommendation System")

# Load model and encoder
model_path = "artifacts/crop_recommendation_model.pkl"
encoder_path = "artifacts/label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    st.error("Model or encoder not found. Please train the model first.")
else:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Weather input from user
    st.subheader("Enter Weather and Soil Conditions")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
    P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
    K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

    if st.button("Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop = encoder.inverse_transform(prediction)[0]
        st.success(f"✅ Recommended Crop: **{crop.upper()}**")
