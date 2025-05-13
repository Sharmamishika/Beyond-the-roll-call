import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Load trained embeddings
with open("trained_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

stored_embeddings = np.array(data["embeddings"])
stored_names = np.array(data["names"])
stored_rolls = np.array(data["roll_numbers"])

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(stored_names)

test_folder = r"C:\Users\Silky\OneDrive\Desktop\MINOR 2\TestImages"

y_true = []
y_pred = []

threshold = 10  # Euclidean distance threshold

for file in os.listdir(test_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, file)

        try:
            embedding_obj = DeepFace.represent(img_path=path, model_name="Facenet512", enforce_detection=False)
            test_embedding = np.array(embedding_obj[0]["embedding"])

            
            distances = np.linalg.norm(stored_embeddings - test_embedding, axis=1)

            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            pred_name = "unknown"
            if min_dist < threshold:
                pred_name = stored_names[min_idx]

            parts = file.split("_")
            true_name = os.path.splitext(parts[1])[0] if len(parts) >= 2 else "unknown"

            y_true.append(true_name)
            y_pred.append(pred_name)

            print(f"✅ Tested {file}: True={true_name}, Pred={pred_name}")

        except Exception as e:
            print(f"⚠️ Could not process {file}: {e}")

# Classification Metrics
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=label_encoder.classes_))

accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm_labels = sorted(set(y_true + y_pred))
cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

#  Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
