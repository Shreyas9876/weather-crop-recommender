from src.data.load_data import load_crop_data
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

#Load data
df = load_crop_data()

# 1.Correlation HeatMap
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("🔎 Correlation between features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# 2. Distribution of each feature
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in features:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"📊 Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"outputs/dist_{col}.png")
    plt.close()

print("✅ EDA plots saved in /outputs/")