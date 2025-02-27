import os
import whisper
from moviepy import VideoFileClip
import nltk
import librosa
import numpy as np  
import pandas as pd
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer  # Make sure to import this

# Set the path to the FFmpeg binary
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"  # Replace with the correct path to your FFmpeg bin folder

# Step 1: Extract audio from video
video_file = "The Marías.mp4"
audio_file = "extracted_audio.wav"

try:
    print("Extracting audio from video...")
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)
    print("Audio extraction complete.")
except Exception as e:
    print(f"Error extracting audio: {e}")
    exit()

# Step 2: Load the Whisper model
try:
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit()

# Step 3: Transcribe the audio
try:
    print("Transcribing audio...")
    transcription = model.transcribe(audio_file)
    print("Transcription complete.")
except Exception as e:
    print(f"Error transcribing audio: {e}")
    exit()

# Step 4: Extract and print the transcribed text
transcribed_text = transcription["text"]
print("Transcribed Text:", transcribed_text)

# Step 5: Save the transcription to a file
try:
    with open("transcription.txt", "w") as file:
        file.write(transcribed_text)
    print("Transcription saved to transcription.txt.")
except Exception as e:
    print(f"Error saving transcription: {e}")

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(transcribed_text)

print("Sentiment Scores:", sentiment_scores)

# To interpret the sentiment:
compound_score = sentiment_scores['compound']
if compound_score >= 0.05:
    print("Overall sentiment: Positive")
elif compound_score <= -0.05:
    print("Overall sentiment: Negative")
else:
    print("Overall sentiment: Neutral")

# Extract pitch, intensity, and speech rate
y, sr = librosa.load(audio_file)

# Pitch extraction using YIN method (corrected)
pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300)  # Pitch extraction with voicing information

# Extract intensity (RMS energy)
intensity = librosa.feature.rms(y=y).mean()  # RMS energy as intensity

# Extract speech rate using beat detection (Note: Not strictly speech rate but rather tempo)
speech_rate = librosa.beat.tempo(y=y, sr=sr)[0]  # Extract rate of speech (tempo)
 # Extract rate of speech (tempo)

# Output the results
print(f"Pitch: {np.nanmean(pitch):.2f} Hz")  # Using np.nanmean to avoid NaN values
print(f"Intensity: {intensity:.2f}")
print(f"Speech Rate (Tempo in BPM): {speech_rate:.2f}")




df = pd.read_csv("trending_videos.csv")
print(df.columns)
features = []
labels = []

for index, row in df.iterrows():
    y, sr = librosa.load(row["file_path"])
    pitch = librosa.yin(y, 50, 300).mean()
    intensity = librosa.feature.rms(y=y).mean()
    features.append([pitch, intensity])
    labels.append(row["emotion"])

X = np.array(features)
y = np.array(labels)