# FINAL CLASSIFIER CODE USED FOR TRAINING THE MODEL WITH 1500 TRAINING IMAGES
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import joblib

# Load the training features
h5_path = r"/Users/vidushiyaksh/Desktop/YEAR4/Computer Vision/Coursework3/vs_cnn_train_features.h5"

with h5py.File(h5_path, "r") as f:
    features = f["features"][:]  # Shape: (1500, 24576)
    labels = f["labels"][:]      # Shape: (1500,) — integers from 0 to 14


# Tried 95 % variance PCA giving mean accuracy accross 5 fold as 85.67%
# With 99%, mean accuracy is mean validation accuracy across 5 folds 86.87%
classifier = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    SVC(kernel='linear')
)

#Train with the full training data set 
classifier.fit(features, labels)

#Save the trained model
joblib.dump(classifier, "svm_pca_model.joblib")

print("Final model trained and saved as 'svm_pca_model.joblib'")
