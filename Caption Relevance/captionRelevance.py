# import requests
# import json
# import pandas as pd 
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer, util
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# API_KEY = API_KEY
# BASE_URL = 'https://www.googleapis.com/youtube/v3/videos'

# params = {
#     'part': 'snippet',
#     'chart': 'mostPopular',
#     'regionCode': 'US',
#     'maxResults': 50,
#     'key': API_KEY
# }

# # 🔹 Fetch YouTube Trending Videos
# response = requests.get(BASE_URL, params=params)

# # 🔹 Handle API Response
# try:
#     data = response.json()
# except json.JSONDecodeError:
#     print("Error: Unable to decode API response.")
#     exit()

# # 🔹 Check for API Errors
# if 'error' in data:
#     print(f"Error {data['error']['code']}: {data['error']['message']}")
#     exit()

# # 🔹 Process Video Metadata
# if 'items' in data:
#     video_metadata = []
#     for item in data['items']:
#         title = item['snippet']['title']
#         description = item['snippet']['description']
#         tags = item['snippet'].get('tags', [])
#         category = item['snippet']['categoryId']
#         video_metadata.append({'title': title, 'description': description, 'tags': ', '.join(tags), 'category': category})

#     # Convert to Pandas DataFrame
#     df = pd.DataFrame(video_metadata)

#     # Save DataFrame as CSV file
#     df.to_csv('trending_videos.csv', index=False, encoding='utf-8')

#     print("Trending videos saved to 'trending_videos.csv'")
# else:
#     print("Error: 'items' key not found in API response.")
#     exit()
# # 🔹 TF-IDF Analysis (Only on Titles to Match Label Count)
# titles = [item['title'] for item in video_metadata]

# vectorizer = TfidfVectorizer(stop_words='english')
# tf_idf_matrix = vectorizer.fit_transform(titles)  # **Only titles, not descriptions**

# # 🔹 Sentence Similarity Analysis (Caption Relevance)
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# trend_keywords = ['viral', 'challenge', 'reaction', 'new music', 'top trends']

# similarity_scores = []
# for video in video_metadata:
#     sentence_embedding = model.encode(video['title'], convert_to_tensor=True)
#     keyword_embeddings = model.encode(trend_keywords, convert_to_tensor=True)
#     similarity = util.pytorch_cos_sim(sentence_embedding, keyword_embeddings).mean().item()
#     similarity_scores.append(similarity)

# print("Average similarity to trending keywords:", sum(similarity_scores) / len(similarity_scores))

# # Model Training

# # 🔹 Prepare Dataset for Machine Learning
# X = tf_idf_matrix.toarray()  # Convert TF-IDF features to array
# y = [1 if similarity_scores[i] > 0.05 else 0 for i in range(len(titles))]  # Adjusted threshold to ensure both labels

# # 🔹 Check for Class Distribution
# unique_labels = set(y)
# print("Class distribution in y:", unique_labels)

# # 🔹 Ensure X and y have the same length before training
# print("X shape:", X.shape, "y length:", len(y))

# # 🔹 Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 🔹 Train Logistic Regression Model
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)

# # 🔹 Evaluate Model
# predictions = classifier.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
# print(classification_report(y_test, predictions))


# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# cm = confusion_matrix(y_test, predictions)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()



# # data collected from US region CSV file
# import pandas as pd

# # Load dataset
# df = pd.read_csv("USvideos.csv")

# # Filter only 'Entertainment' category videos (assuming category_id for Entertainment is 24)
# df_entertainment = df[df["category_id"] == 24]

# # Keep only relevant columns and remove duplicates
# df_unique = df_entertainment[["video_id", "title", "description"]].dropna().drop_duplicates()

# # Save cleaned dataset
# df_unique.to_csv("captions_dataset_entertainment.csv", index=False)

# print(f"✅ Created dataset with {len(df_unique)} unique captions from the Entertainment category!")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset
df_captions = pd.read_csv("captions_dataset_entertainment.csv")

# Function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
df_captions["clean_title"] = df_captions["title"].apply(preprocess_text)
df_captions["clean_description"] = df_captions["description"].apply(preprocess_text)

# Save processed dataset
df_captions.to_csv("processed_captions.csv", index=False)

print("✅ Text preprocessing complete! Saved as 'processed_captions.csv'.")