# import pandas as pd

# # Step 1: Load the dataset
# file_path = "USvideos.csv"  # Change this if needed
# df = pd.read_csv(file_path)

# # Step 2: Filter for Entertainment category (category_id = 24)
# df_entertainment = df[df["category_id"] == 24]  # Ensure category_id is an integer

# # Step 3: Save the filtered dataset
# df_entertainment.to_csv("entertainment_videos.csv", index=False)

# print(f"Filtered {len(df_entertainment)} entertainment videos and saved successfully!")


# import pandas as pd

# # Load the filtered dataset
# df = pd.read_csv("entertainment_videos.csv")

# # Keep only the video_id column
# df_video_ids = df[["video_id"]]

# # Save the cleaned file
# df_video_ids.to_csv("video_ids_only.csv", index=False)

# print(f"Saved {len(df_video_ids)} video IDs for audio analysis!")
# import pandas as pd

# # Load the filtered dataset containing video IDs
# df = pd.read_csv("video_ids_only.csv")

# # Remove duplicates and keep only unique video IDs
# df_unique_video_ids = df.drop_duplicates(subset=['video_id'])

# # Save the unique video IDs to a new file
# df_unique_video_ids.to_csv("unique_video_ids.csv", index=False)

# # Print the number of unique video IDs saved
# print(f"Saved {len(df_unique_video_ids)} unique video IDs to 'unique_video_ids.csv'.")


# import pandas as pd
# import os

# # Load video IDs
# df = pd.read_csv("unique_video_ids.csv")

# # Create a directory for downloaded videos
# os.makedirs("videos", exist_ok=True)

# # Loop through each video and download only the audio
# for video_id in df["video_id"]:
#     video_url = f"https://www.youtube.com/watch?v={video_id}"
#     output_path = f"videos/{video_id}.mp4"
    
#     # Download using yt-dlp (best audio only)
#     os.system(f'yt-dlp -f "bestaudio" -o "{output_path}" {video_url}')

# print("Download complete! All videos are saved in the 'videos' folder.")



import pandas as pd
import os
import ffmpeg

# Load video IDs
df = pd.read_csv("unique_video_ids.csv")

# Ensure 'audio' directory exists
os.makedirs("audio", exist_ok=True)

#Loop through downloaded videos and extract audio
# for video_id in df["video_id"]:
#     input_path = f"videos/{video_id}.mp4"
#     output_path = f"audio/{video_id}.wav"
    
#     # ✅ Check if the video file exists before processing
#     if os.path.exists(input_path):
#         try:
#             ffmpeg.input(input_path).output(output_path, format="wav").run(overwrite_output=True)
#             print(f"✅ Extracted audio: {output_path}")
#         except Exception as e:
#             print(f"⚠️ Error extracting audio from {video_id}: {e}")
#     else:
#         print(f"❌ Skipping {video_id}: File not found!")

# print("🎉 Audio extraction process completed!")

import os
import ffmpeg

# Ensure the output folder exists
os.makedirs("audio_mp3", exist_ok=True)

# Loop through all WAV files and convert them to MP3
for file in os.listdir("audio"):
    if file.endswith(".wav"):  # Process only WAV files
        wav_path = os.path.join("audio", file)
        mp3_path = os.path.join("audio_mp3", file.replace(".wav", ".mp3"))

        try:
            # Convert WAV to MP3 (128kbps for good quality)
            ffmpeg.input(wav_path).output(mp3_path, format="mp3", audio_bitrate="128k").run(overwrite_output=True)
            print(f"✅ Converted: {file} → {mp3_path}")
        except Exception as e:
            print(f"⚠️ Error converting {file}: {e}")

print("🎉 All WAV files have been converted to MP3!")











# import requests
# import time

# API_KEY = API_KEY
# SEARCH_URL = "https://www.googleapis.com/youtube/v3/videos"

# # List of regions to fetch trending videos from
# REGIONS = ["US", "GB", "CA", "AU", "IN", "SG", "NZ", "ZA", "IE", "PH"]  # 10 countries

# # Parameters to fetch trending videos
# params = {
#     "part": "snippet",
#     "chart": "mostPopular",  # Fetch trending videos
#     "videoCategoryId": "23",  # Comedy Category
#     "maxResults": 50,  # Maximum per request
#     "key": API_KEY
# }

# video_ids = set()  # Use a set to avoid duplicates
# total_videos_needed = 2000

# # Fetch trending videos from multiple countries
# for region in REGIONS:
#     params["regionCode"] = region
#     next_page_token = None

#     while len(video_ids) < total_videos_needed:
#         if next_page_token:
#             params["pageToken"] = next_page_token

#         response = requests.get(SEARCH_URL, params=params)
#         data = response.json()

#         if "items" in data:
#             for item in data["items"]:
#                 video_ids.add(item["id"])

#         next_page_token = data.get("nextPageToken")
#         if not next_page_token:
#             break

#         print(f"Collected {len(video_ids)} video URLs so far...")
#         time.sleep(2)

#     if len(video_ids) >= total_videos_needed:
#         break

# # Save URLs to a text file
# with open("trending_comedy_videos.txt", "w") as f:
#     for vid in list(video_ids)[:total_videos_needed]:
#         f.write(f"https://www.youtube.com/watch?v={vid}\n")

# print(f"✅ Successfully saved {len(video_ids)} trending comedy video URLs.")