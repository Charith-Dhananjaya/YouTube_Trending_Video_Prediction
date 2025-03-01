import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("USvideos.csv")

# Select engagement features
df_engagement = df[["video_id", "views", "likes", "dislikes", "comment_count"]].dropna()

# Normalize engagement metrics (scale between 0-1)
scaler = MinMaxScaler()
df_engagement[["views", "likes", "dislikes", "comment_count"]] = scaler.fit_transform(df_engagement[["views", "likes", "dislikes", "comment_count"]])

# Save processed dataset
df_engagement.to_csv("engagement_metrics.csv", index=False)

print("✅ Engagement metrics extracted and normalized! Saved as 'engagement_metrics.csv'.")
