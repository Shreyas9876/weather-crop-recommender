🌾 Weather-Based Crop Recommendation System
A machine learning-powered web application that recommends the most suitable crop to cultivate based on real-time soil and weather conditions such as Nitrogen, Phosphorus, Potassium levels, temperature, humidity, pH, and rainfall.


Features

Machine learning model trained on crop recommendation dataset.
Accurate crop predictions with 99%+ accuracy.
Interactive web app built with Streamlit.
Modular code structure with separate data ingestion, preprocessing, model training, and prediction scripts.
Deployable to Streamlit Cloud.
End-to-end working pipeline from dataset to deployed model.


Project Structure

weather-crop-recommender/
├── artifacts/
│   ├── crop_recommendation_model.pkl
│   ├── label_encoder.pkl
│   └── Crop_recommendation.csv
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── models/
│   │   └── train_model.py
│   ├── preprocessing.py
│   └── predict.py
├── app/
├── streamlit_app.py
├── requirements.txt
├── README.md


Dataset

Source: Kaggle Crop Recommendation Dataset

Records: 2200
Features:
N - Nitrogen
P - Phosphorus
K - Potassium
temperature (in °C)
humidity (in %)
ph - Soil pH value
rainfall (in mm)
label - Crop type (Target)
🔧 Setup Instructions
🔗 Clone Repository
bash
Copy
Edit
git clone https://github.com/Shreyas9876/weather-crop-recommender.git
cd weather-crop-recommender

Deployed App
https://weather-crop-recommender-fnbsowfkezrcsonted4mrr.streamlit.app/

