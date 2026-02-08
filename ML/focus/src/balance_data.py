"""
Script to balance the dataset by oversampling minority classes.
This creates synthetic balance by duplicating rows from underrepresented classes.
"""

import os
import pandas as pd
import numpy as np

# Load the generated labels
df = pd.read_csv("data/labels.csv")

print("BEFORE BALANCING:")
print(df['label'].value_counts())
print()

# Get counts
counts = df['label'].value_counts()
max_count = counts.max()

# Oversample each class to match the maximum
balanced_dfs = []
for label in df['label'].unique():
    label_df = df[df['label'] == label]
    current_count = len(label_df)
    
    if current_count < max_count:
        # Calculate how many times to duplicate
        times_to_duplicate = max_count // current_count
        remainder = max_count % current_count
        
        # Duplicate the data
        duplicated = pd.concat([label_df] * times_to_duplicate, ignore_index=True)
        
        # Add remaining samples randomly
        if remainder > 0:
            extra = label_df.sample(n=remainder, random_state=42)
            duplicated = pd.concat([duplicated, extra], ignore_index=True)
        
        balanced_dfs.append(duplicated)
    else:
        balanced_dfs.append(label_df)

# Combine all
df_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("AFTER BALANCING:")
print(df_balanced['label'].value_counts())
print()
print(f"Total images: {len(df_balanced)}")

# Save
df_balanced.to_csv("data/labels_balanced.csv", index=False)
print("\nSaved to data/labels_balanced.csv")
