import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the raw data
df = pd.read_csv('data/cleaned_merged_heart_dataset.csv')

# Split the data
# Using random_state=42 ensures our splits are reproducible
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the datasets
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Preprocessing complete: train.csv and test.csv saved to the data/ folder.")
