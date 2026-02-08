import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load original labels
df = pd.read_csv("data/labels.csv")

# Make image paths absolute
df['image_path'] = df['image_path'].apply(lambda x: os.path.abspath(os.path.join(".", x)))

# Map labels
label_map = {"Focused": 0, "Drifting": 1, "Lost": 2}
df['label_idx'] = df['label'].map(label_map)

print("Original counts:")
print(df['label'].value_counts())

# Oversample Focused to balance
focused_df = df[df['label'] == 'Focused']
for _ in range(2):  # Triple the Focused samples
    df = pd.concat([df, focused_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced counts:")
print(df['label'].value_counts())

# Save balanced csv
df.to_csv("data/labels_balanced.csv", index=False)
print("Saved to data/labels_balanced.csv")

# Split for reference
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label_idx'], random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label_idx'], random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")