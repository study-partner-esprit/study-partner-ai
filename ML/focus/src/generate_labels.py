"""
Script to generate labels.csv from the original data folders.

Folder structure:
- data/raw/Engaged/engaged/      → Focused
- data/raw/Engaged/confused/     → Drifting
- data/raw/Engaged/frustrated/   → Lost
- data/raw/Not engaged/bored/    → Drifting
- data/raw/Not engaged/drowsy/   → Lost
- data/raw/Not engaged/Looking Away/ → Lost
"""

import os
import pandas as pd

# Define mapping from folder to label
folder_to_label = {
    "Engaged/engaged": "Focused",
    "Engaged/confused": "Drifting",
    "Engaged/frustrated": "Lost",
    "Not engaged/bored": "Drifting",
    "Not engaged/drowsy": "Lost",
    "Not engaged/Looking Away": "Lost",
}

data_dir = "data/raw"
entries = []

for folder, label in folder_to_label.items():
    folder_path = os.path.join(data_dir, folder)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(data_dir, folder, file)
                entries.append({"image_path": image_path, "label": label})

# Create DataFrame
df = pd.DataFrame(entries)

# Show counts per label
print("=" * 40)
print("DATA SUMMARY")
print("=" * 40)
print(f"Total images: {len(df)}")
print()
print("Counts per label:")
print(df['label'].value_counts())
print()
print("Counts per source folder:")
df['source'] = df['image_path'].apply(lambda x: '/'.join(x.split('/')[2:4]))
print(df.groupby(['source', 'label']).size())
print()

# Save to CSV
df[['image_path', 'label']].to_csv("data/labels.csv", index=False)
print("Saved to data/labels.csv")
