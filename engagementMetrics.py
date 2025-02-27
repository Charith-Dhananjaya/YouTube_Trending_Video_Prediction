import requests
import json
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

API_KEY = "AIzaSyD11o7fbZQAgeDzl1YxumbXHHy22Wq_76U"
VIDEO_ID = "dQw4w9WgXcQ"  # Example YouTube video ID
BASE_URL = "https://www.googleapis.com/youtube/v3/videos"

params = {
    "part": "statistics",
    "id": VIDEO_ID,
    "key": API_KEY
}

response = requests.get(BASE_URL, params=params)
data = response.json()

if "items" in data:
    video_stats = data["items"][0]["statistics"]
    
    # Print the data
    print("Likes:", video_stats.get("likeCount", 0))
    print("Comments:", video_stats.get("commentCount", 0))
    print("Views:", video_stats.get("viewCount", 0))
    print("Shares (Estimate):", int(video_stats.get("viewCount", 0)) * 0.02)  # Approximation

    # Prepare data for CSV
    video_data = {
        "video_id": VIDEO_ID,
        "likes": video_stats.get("likeCount", 0),
        "comments": video_stats.get("commentCount", 0),
        "views": video_stats.get("viewCount", 0),
        "shares_estimate": int(video_stats.get("viewCount", 0)) * 0.02  # Approximation
    }

    # Save to CSV
    csv_file = "video_statistics.csv"
    
    # Check if the file exists to write headers
    try:
        with open(csv_file, mode="r") as file:
            pass
    except FileNotFoundError:
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=video_data.keys())
            writer.writeheader()
    
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=video_data.keys())
        writer.writerow(video_data)

    print(f"Data saved to {csv_file}")

total_views = int(video_stats.get("viewCount", 0))
total_watch_time = total_views * 3.5  # Assume average 3.5 min watch time
video_length = 10  # Assume 10 min video length

retention_rate = (total_watch_time / (total_views * video_length)) * 100
print(f"Estimated Retention Rate: {retention_rate:.2f}%")


params = {
    "part": "snippet",
    "videoId": VIDEO_ID,
    "key": API_KEY,
    "maxResults": 100
}

response = requests.get("https://www.googleapis.com/youtube/v3/commentThreads", params=params)
comments_data = response.json()

comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in comments_data["items"]]


sia = SentimentIntensityAnalyzer()

comment_sentiments = [sia.polarity_scores(comment)["compound"] for comment in comments]

# Calculate overall comment sentiment
average_sentiment = sum(comment_sentiments) / len(comment_sentiments)
print(f"Average Comment Sentiment Score: {average_sentiment:.2f}")



vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(comments)

# Calculate similarity between comments
similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

# Count highly similar comments (possible spam)
spam_count = (similarity_matrix > 0.9).sum() - len(comments)
print(f"Possible Spam Comments: {spam_count}")


