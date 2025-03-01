import pandas as pd

# Load dataset with video details
df = pd.read_csv("USvideos.csv")

# Keep only relevant columns
df_captions = df[["video_id", "title", "description"]].dropna()

# Save cleaned dataset
df_captions.to_csv("captions_dataset.csv", index=False)

print(f"✅ Created dataset with {len(df_captions)} captions for analysis!")
