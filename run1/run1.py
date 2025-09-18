#!/usr/bin/env python3
"""
scene_classification_tiny_knn.py

Load 15-scene dataset, extract tiny-image features, tune a KNN
classifier via 5-fold cross-validation, test on held-out images,
and compute precision.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def tiny_image_feature(img, output_size=16):
    """
    Compute a tiny-image feature descriptor for a grayscale image.
    Steps:
      1. Center-crop to square
      2. Resize to (output_size × output_size)
      3. Flatten to a 1D vector
      4. Zero-mean and unit-length normalize
    """
    h, w = img.shape

    # Center-crop the longer dimension
    if h < w:
        start = (w - h) // 2
        img_cropped = img[:, start:start+h]
    else:
        start = (h - w) // 2
        img_cropped = img[start:start+w, :]

    # Resize to fixed small size
    img_resized = cv2.resize(img_cropped,
                             (output_size, output_size),
                             interpolation=cv2.INTER_AREA)

    # Flatten into vector of floats
    feat = img_resized.flatten().astype(np.float32)

    # Subtract mean to center data
    feat -= np.mean(feat)

    # Divide by norm to get unit length (if not all zeros)
    norm = np.linalg.norm(feat)
    if norm > 1e-5:
        feat /= norm

    return feat


def load_dataset(image_dir):
    """
    Walk through subdirectories of image_dir, read all
    .jpg/.jpeg/.png images in grayscale, and return
    (list_of_images, list_of_labels).
    """
    images = []
    labels = []
    categories = sorted(os.listdir(image_dir))

    for category in categories:
        folder = os.path.join(image_dir, category)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(category)

    return images, np.array(labels)


def extract_features(img_list, feature_fn, **kwargs):
    """
    Given a list of images and a feature extraction function,
    return a NumPy array of feature vectors.
    """
    n = len(img_list)
    # Assume feature length from one example
    dummy = feature_fn(img_list[0], **kwargs)
    feats = np.zeros((n, dummy.size), dtype=np.float32)

    for i, img in enumerate(img_list):
        feats[i, :] = feature_fn(img, **kwargs)

    return feats


def find_best_k(feats, labels, k_values, cv_folds=5):
    """
    Perform cross-validation for each k in k_values,
    return (best_k, list_of_mean_accuracies).
    """
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, feats, labels,
                                 cv=cv_folds, scoring='accuracy')
        accuracies.append(scores.mean())
        print(f"K={k}: Mean CV Accuracy = {scores.mean():.4f}")

    best_idx = np.argmax(accuracies)
    return k_values[best_idx], accuracies


def load_test_images(test_dir):
    """
    Read and sort test images named like '0.jpg', '1.jpg', ...
    Returns (images_list, filenames_list).
    """
    files = sorted(os.listdir(test_dir),
                   key=lambda x: int(os.path.splitext(x)[0]))
    images = []
    fnames = []

    for fname in files:
        if fname.lower().endswith('.jpg'):
            path = os.path.join(test_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                fnames.append(fname)

    return images, fnames


def save_predictions(filenames, preds, out_file):
    """
    Write lines "filename label" to out_file.
    """
    with open(out_file, 'w') as f:
        for fn, p in zip(filenames, preds):
            f.write(f"{fn} {p}\n")


def load_predictions(file_path):
    """
    Read "filename label" pairs from file, return dict.
    """
    d = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d


def evaluate(pred_file, truth_file):
    """
    Compare predicted vs. ground truth labels
    (case-insensitive), return (correct, total, precision).
    """
    pred = load_predictions(pred_file)
    truth = load_predictions(truth_file)

    total = len(truth)
    correct = sum(1 for fn, gt in truth.items()
                  if fn in pred and pred[fn].lower() == gt.lower())

    precision = correct / total if total > 0 else 0.0
    return correct, total, precision


def plot_cv_results(k_vals, accuracies, best_k, best_acc):
    """
    Plot K vs. CV accuracy, highlight the best point.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, accuracies, label='5-Fold CV Acc.')
    plt.scatter([best_k], [best_acc], label=f'Best K={best_k}\nAcc={best_acc:.4f}')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('KNN Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.xlim([min(k_vals)-1, max(k_vals)+1])
    plt.ylim([min(accuracies)-0.01, max(accuracies)+0.01])
    plt.show()


def main():
    # === Parameters and paths ===
    train_dir = "/Users/nadiahaque/Library/CloudStorage/OneDrive-UniversityofSouthampton/SustAI/Semester_2/COMP6223/Assignments/Assignment3/training"
    test_dir  = "/Users/nadiahaque/Library/CloudStorage/OneDrive-UniversityofSouthampton/SustAI/Semester_2/COMP6223/Assignments/Assignment3/testing"
    answers_file = "answer.txt"
    pred_file    = "run1.txt"
    k_vals = list(range(3, 100, 2))

    # === 1) Load and report training data ===
    train_imgs, train_lbls = load_dataset(train_dir)
    print(f"[INFO] Loaded {len(train_imgs)} train images "
          f"across {len(np.unique(train_lbls))} classes.")

    # === 2) Extract tiny-image features ===
    train_feats = extract_features(train_imgs,
                                   tiny_image_feature,
                                   output_size=16)
    print(f"[INFO] Extracted features: {train_feats.shape}")

    # === 3) Cross-validate to find best K ===
    print("[INFO] Performing 5-fold CV to select K...")
    best_k, cv_acc = find_best_k(train_feats, train_lbls, k_vals)
    print(f"[INFO] Best K = {best_k}, CV Acc = {max(cv_acc):.4f}")

    # === 4) Retrain KNN on full training set ===
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_feats, train_lbls)
    print("[INFO] Final KNN model trained on full data.")

    # === 5) Load and featurize test images ===
    test_imgs, test_names = load_test_images(test_dir)
    print(f"[INFO] Loaded {len(test_imgs)} test images.")
    test_feats = extract_features(test_imgs,
                                  tiny_image_feature,
                                  output_size=16)

    # === 6) Predict and save results ===
    preds = knn.predict(test_feats)
    save_predictions(test_names, preds, pred_file)
    print(f"[INFO] Predictions saved to {pred_file}")

    # === 7) Evaluate precision against ground truth ===
    correct, total, prec = evaluate(pred_file, answers_file)
    print(f"[INFO] Precision: {prec:.4f} ({correct}/{total} correct)")

    # === 8) Plot CV accuracy vs K ===
    plot_cv_results(k_vals, cv_acc, best_k, max(cv_acc))


if __name__ == "__main__":
    main()