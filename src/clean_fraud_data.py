# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Load the dataset
df = pd.read_csv('../data/processed/fraud_data_with_country.csv')

# 3. Preview the dataset
print("Initial preview:")
print(df.head())

# 4. Data Overview
print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 5. Handling Missing Values
missing = df.isnull().sum()
missing = missing[missing > 0]

if not missing.empty:
    print("\nMissing Values Detected:")
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percent': missing_percent
    })
    print(missing_df)
else:
    print("\nNo missing values found.")


# 6. Data Cleaning

# Remove duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"\nRemoved {initial_shape[0] - df.shape[0]} duplicate rows.")

# Convert datetime columns (if not already converted)
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])

# Final Data Overview
print("\nFinal Data Info:")
print(df.info())
print("Final shape:", df.shape)

# 7. Save Cleaned Data
df.to_csv('../data/processed/cleaned_fraud_data.csv', index=False)
print("\nCleaned data saved to ../data/processed/cleaned_fraud_data.csv")
