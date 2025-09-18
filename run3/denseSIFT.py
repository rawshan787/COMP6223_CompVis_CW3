# Traditional method implementation of Dense SIFT for comparison with our chosen method of AlexNet + PCA.
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

# Training images directory
src_dir = "/Users/vidushiyaksh/Desktop/YEAR4/Computer Vision/Coursework3/training"

# Dense SIFT feature extraction function
def dense_sift_feat_ext(image):
    sift = cv2.SIFT_create()

    # Define dense keypoints at fixed step size of every 8 pixels
    step_size = 8

    keypoint = [cv2.KeyPoint(x, y, step_size) 
          for y in range(0, image.shape[0], step_size) 
          for x in range(0, image.shape[1], step_size)]

    # Compute SIFT descriptors at the specified keypoints
    _, descriptor = sift.compute(image, keypoint)

    # if no descriptor found handle by filling with zeros.
    if descriptor is None:
        return np.zeros(128)
    
    # Flatten the extracted features into 1D vector
    return descriptor.flatten()

# X is the extracted feature array and y is the image label name
X = []
y = []

# Loop through all training folder classes and extract the features
for label in os.listdir(src_dir):
    class_dir = os.path.join(src_dir, label)
    if not os.path.isdir(class_dir):
        continue

    
    for filename in tqdm(os.listdir(class_dir), desc=f"Class: {label}"):
        if not filename.lower().endswith('.jpg'):
            continue

        # Build the full path to the image file
        image_path = os.path.join(class_dir, filename)

        # Read the image directly grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Skip the file if it failed to load
        if img is None:
            print(f"Failed to load {image_path}.")
            continue

        # Extract Dense SIFT features for the current image
        feat = dense_sift_feat_ext(img)

        # Save the feature vector and its corresponding class label
        X.append(feat)
        y.append(label)

# Convert extracted features to numpy arrays correctly
# Since images can have different numbers of descriptors, do zero-padding to align all vectors
max_len = max(len(feat) for feat in X)
X = np.array([np.pad(feat, (0, max_len - len(feat))) for feat in X])
y = np.array(y)

print(f"Feature extraction complete.")

#TESTING VALIDATION RESULTS HERE
classifier = SVC(kernel="linear")

# Stratified 5 fold split, exactly the same as AlexNet + PCA version
kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Calculate score for accuracy
scores = cross_val_score(classifier, X, y, cv=kfold_split, scoring='accuracy')

# OUTPUT RESULTS
print("Accuracy of cross validation:", scores)
print(f"Mean accuracy over 5 folds: {np.mean(scores):.4f}")