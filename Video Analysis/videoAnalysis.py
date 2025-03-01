import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1️⃣ Extract Frames from a Video
# =====================================
def extract_frames(video_path, frame_interval=30):
    os.makedirs("video_frames", exist_ok=True)  # Create directory for frames

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            frame_filename = f"video_frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"✅ Saved: {frame_filename}")
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print("\n✅ Frame extraction complete!\n")

# =====================================
# 2️⃣ Analyze Color Distribution in a Frame
# =====================================
def analyze_color_distribution(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Cannot read image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    plt.figure(figsize=(8, 4))
    plt.plot(r_hist, color='red', label='Red')
    plt.plot(g_hist, color='green', label='Green')
    plt.plot(b_hist, color='blue', label='Blue')
    plt.title("Color Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# =====================================
# 3️⃣ Calculate Motion Magnitude in a Video
# =====================================
def calculate_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("❌ Error: Cannot read video file")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_motion = 0
    frame_count = 0

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        motion_magnitude = np.linalg.norm(flow)
        total_motion += motion_magnitude
        frame_count += 1
        prev_gray = next_gray

        if frame_count % 1000 == 0:  # Print progress every 100 frames
            print(f"✅ Processed {frame_count} frames...")

    cap.release()

    avg_motion = total_motion / frame_count if frame_count > 0 else 0
    print(f"\n✅ Final Average Motion Magnitude: {avg_motion:.2f}\n")

# =====================================
# Run the Module (Example Usage)
# =====================================
if __name__ == "__main__":
    video_file = "videoplayback.mp4"  # Change this to your video file path

    print("\n🎬 Starting Video Analysis...\n")
    
    # Extract frames
    extract_frames(video_file)

    # Analyze a sample frame
    analyze_color_distribution("video_frames/frame_30.jpg")  # Change frame if needed

    # Compute motion
    calculate_motion(video_file)

    print("\n✅ Video Analysis Completed!\n")



from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Using a pre-trained YOLOv8 model

# Detect objects in an extracted frame
results = model("video_frames/frame_30.jpg", show=True)

# Print detected objects
for result in results:
    for obj in result.boxes.data:
        print(f"Object: {model.names[int(obj[5])]}, Confidence: {obj[4]:.2f}")


import pandas as pd

data = {
    "motion_magnitude": [2.5, 4.8, 1.2, 6.1, 3.3],  # Example motion values
    "color_variance": [120.5, 98.3, 145.2, 78.4, 110.1],  # Example color values
    "object_count": [5, 7, 2, 10, 3],  # Number of detected objects
    "video_length": [120, 300, 150, 600, 180],  # Length in seconds
    "is_trending": [1, 1, 0, 1, 0]  # 1 = Trending, 0 = Non-Trending
}

df = pd.DataFrame(data)
df.to_csv("video_analysis_data.csv", index=False)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("video_analysis_data.csv")
X = df.drop(columns=["is_trending"])
y = df["is_trending"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
