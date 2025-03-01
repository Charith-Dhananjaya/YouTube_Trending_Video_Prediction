import pandas as pd
import googleapiclient.discovery
import time

### STEP 1: Extract Entertainment Videos ###
# Load the dataset
file_path = "USvideos.csv"  # Change if needed
df = pd.read_csv(file_path)

# Convert category_id to string (assuming it's stored as int)
df['category_id'] = df['category_id'].astype(str)

# Entertainment category ID (check actual ID in dataset)
entertainment_category_id = "24"  # ID for Entertainment category in YouTube datase

# Filter only entertainment category videos
entertainment_videos = df[df['category_id'] == entertainment_category_id]

# Drop duplicate video IDs
entertainment_videos = entertainment_videos.drop_duplicates(subset=['video_id'])

# Save processed entertainment videos
entertainment_file = "entertainment_videos.csv"
entertainment_videos.to_csv(entertainment_file, index=False)

print(f"Entertainment videos data saved to {entertainment_file}")


### STEP 2: Fetch Comments for Each Video ID ###
# YouTube API setup
API_KEY = "AIzaSyD11o7fbZQAgeDzl1YxumbXHHy22Wq_76U"  # Replace with your actual API key
api_service_name = "youtube"
api_version = "v3"
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=API_KEY)

# Function to fetch comments for a given video ID
def get_video_comments(video_id, max_comments=50):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"],
                "published_at": comment["publishedAt"]
            })
    except Exception as e:
        print(f"Error fetching comments for {video_id}: {e}")
        time.sleep(1)  # Prevent API rate limit issues

    return comments

# Collect comments for all entertainment videos
video_ids = entertainment_videos['video_id'].unique().tolist()
all_comments = []
for i, vid in enumerate(video_ids):
    print(f"Fetching comments for video {i+1}/{len(video_ids)}: {vid}")
    comments = get_video_comments(vid, max_comments=50)
    all_comments.extend(comments)
    time.sleep(1)  # Add a delay to avoid API rate limi

# Convert to DataFrame and save
comments_df = pd.DataFrame(all_comments)
comments_file = "entertainment_video_comments.csv"
comments_df.to_csv(comments_file, index=False)

print(f"Comments saved to {comments_file}")


### STEP 3: Merge Processed Videos with Comments ###
# Load processed video and comment data
entertainment_videos = pd.read_csv(entertainment_file)
comments_df = pd.read_csv(comments_file)

# Merge comments with video data using 'video_id'
final_df = entertainment_videos.merge(comments_df, on="video_id", how="left")

# Save final dataset
final_file = "entertainment_videos_with_comments.csv"
final_df.to_csv(final_file, index=False)

print(f"Final dataset with comments saved to {final_file}")
