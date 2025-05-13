import os
import cv2
import numpy as np

# Paths
source_folder = r"C:\Users\Silky\OneDrive\Desktop\MINOR 2\Faces\Faces"
test_folder = r"C:\Users\Silky\OneDrive\Desktop\MINOR 2\TestImages"

os.makedirs(test_folder, exist_ok=True)

def augment_image(image):
    """Apply transformations to create variations."""
    augmented = []

    # Original resized
    resized = cv2.resize(image, (160, 160))
    augmented.append(resized)

    # Slight rotation
    center = (80, 80)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated = cv2.warpAffine(resized, M, (160, 160))
    augmented.append(rotated)

    # Brightness increase
    bright = cv2.convertScaleAbs(resized, alpha=1.2, beta=30)
    augmented.append(bright)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    augmented.append(blurred)

    return augmented

# Loop through each image in training folder
for file in os.listdir(source_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            augmented_imgs = augment_image(img)
            base_name = os.path.splitext(file)[0]
            for i, aug in enumerate(augmented_imgs):
                new_filename = f"{base_name}_aug{i+1}.jpg"
                cv2.imwrite(os.path.join(test_folder, new_filename), aug)

print(f"✅ Test images saved to: {test_folder}")
