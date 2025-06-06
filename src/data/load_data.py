import pandas as pd

def load_crop_data(path: str = "/workspaces/weather-crop-recommender/artifacts/Crop_recommendation.csv") -> pd.DataFrame:
    """
    Load crop recommendation dataset.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()
