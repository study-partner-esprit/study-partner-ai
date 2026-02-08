import os
import pandas as pd

# Load existing labels
df = pd.read_csv('data/labels.csv')

# Filter out rows where image_path does not exist
df = df[df['image_path'].apply(os.path.exists)]

# Save back
df.to_csv('data/labels.csv', index=False)

print(f"Cleaned labels.csv, now has {len(df)} entries")
print(df['label'].value_counts())