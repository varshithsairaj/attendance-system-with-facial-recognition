from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# Define file paths
embeddingFile = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\output\embeddings.pickle"
recognizerFile = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\output\recognizer.pickle"
labelEncFile = r"C:\Users\asus\Documents\FACE attendance[1]\attendance-system-master\output\le.pickle"

# Check if embeddings file exists
if not os.path.exists(embeddingFile):
    print("Error: Embeddings file not found! Run embedding extraction first.")
    exit()

print("🔹 Loading face embeddings...")
with open(embeddingFile, "rb") as f:
    data = pickle.load(f)

# Ensure embeddings are available
if "embeddings" not in data or "names" not in data:
    print("Error: Invalid embeddings file format!")
    exit()

print("🔹 Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

print("🔹 Training SVM model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Save trained recognizer
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)

# Save label encoder
with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)

print("✅ Model training completed and saved successfully!")
