import os
import pickle
from deepface import DeepFace

faces_folder = r"C:\Users\Silky\OneDrive\Desktop\MINOR 2\Faces\Faces"  
output_file = "trained_embeddings.pkl"  

embeddings = []
names = []
roll_numbers = []

# Read all face images
if not os.path.exists(faces_folder):
    print(f"Error: Folder not found -> {faces_folder}")
    exit()

for file_name in os.listdir(faces_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            path = os.path.join(faces_folder, file_name)

            # Split file name by underscore
            parts = file_name.split('_')
            if len(parts) >= 2:
                roll_no = parts[0]
                name = os.path.splitext(parts[1])[0]
            else:
                print(f"Skipping file with unexpected name format: {file_name}")
                continue

            # Get facial embedding
            embedding_obj = DeepFace.represent(img_path=path, model_name="Facenet512", enforce_detection=False)
            embedding = embedding_obj[0]["embedding"]

            embeddings.append(embedding)
            roll_numbers.append(roll_no)
            names.append(name)

            print(f"✅ Processed: {name} ({roll_no})")

        except Exception as e:
            print(f"⚠️ Could not process {file_name}: {e}")

# Save all embeddings to a pickle file
data = {
    "embeddings": embeddings,
    "roll_numbers": roll_numbers,
    "names": names
}

with open(output_file, "wb") as f:
    pickle.dump(data, f)

print(f"\n🎯 Training complete! Embeddings saved to '{output_file}'")
