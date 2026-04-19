import pandas as pd

# Load dataset with correct encoding
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep only required columns (fix for common dataset format)
df = df[['v1', 'v2']]

# Rename columns to standard names
df.columns = ['label', 'message']

# Show sample data
print("Sample Data:\n")
print(df.head())

# Show dataset info
print("\nDataset Info:\n")
print(df.info())

# Show class distribution
print("\nClass Distribution:\n")
print(df['label'].value_counts())