from src.data.load_data import load_crop_data

df = load_crop_data()

print("/n First 5 rows")
print(df.head())

print("\n🔹 Data Summary:")
print(df.describe())

print("\n🔹 Class Distribution:")
print(df['label'].value_counts())

