# FINAL CLASSIFIER CODE FOR VALIDATING THE MODEL
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

# Load the feature vectors and labels
feature_path = r"/Users/vidushiyaksh/Desktop/YEAR4/Computer Vision/Coursework3/vs_cnn_train_features.h5"

#
with h5py.File(feature_path, "r") as f:
    features = f["features"][:]  # Get the feature vectors of each image
    labels = f["labels"][:]      # Get the labels of each image


# With 99%, mean accuracy is mean validation accuracy across 5 folds 86.87%
# Normalisation with mean = 0 and standard deviation=1 and PCA with 99% variance mixed with Linear SVM classifier
classifier = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.99),
    SVC(kernel='linear')
)

#split into 5 folds with equal distribution of class, 42 means reproducable results
kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#split into 5 folds
#for each fold train the model on 4/5th of the data with 1/5th as validation
#return an accuracy score
scores = cross_val_score(classifier, features, labels, cv=kfold_split, scoring='accuracy')

#Print results
print("\nCross-validation scores (accuracy):", scores)
print(f"Mean accuracy over 5 folds: {np.mean(scores):.4f}")
