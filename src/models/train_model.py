import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

from src.data.load_data import load_crop_data

def train_model():
    df = load_crop_data()

    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split( X, y_encoded, test_size = 0.2, random_state = 42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model. predict(X_test)
    print("✅ Clasification Report: \n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model and label encoder
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/crop_model.pkl")
    joblib.dump(encoder, "artifacts/label_encoder.pkl")
    print("\n✅ Model and encoder saved in /artifacts")

if __name__=="__main__":
    train_model()