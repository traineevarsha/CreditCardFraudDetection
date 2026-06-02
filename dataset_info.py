import pandas as pd

# Load dataset
df = pd.read_csv("data/fraudTrain.csv")

print("\n========== DATASET OVERVIEW ==========\n")

# Shape
print("Dataset Shape:")
print(df.shape)

# Columns
print("\nDataset Columns:")
print(df.columns.tolist())

# Info
print("\nDataset Info:")
print(df.info())

# Statistics
print("\nDataset Statistics:")
print(df.describe())

# Fraud Distribution
print("\nFraud Distribution:")
print(df["is_fraud"].value_counts())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())