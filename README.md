# COMP6223 Coursework 3

This project explores multiple approaches to scene recognition on the 15-scene dataset, advancing from traditional to deep learning-based methods. We begin with a K-Nearest Neighbours classifier using tiny image features, achieving 22.01% accuracy. Next, a Bag-of-Visual-Words model with a linear classifier improves accuracy to 70% by leveraging local image patches and clustering. Finally, a modified Deep Visually Sensitive CNN (VS-CNN) incorporates context-based saliency detection and AlexNet features, followed by PCA and SVM classification, achieving 86.87% accuracy.

---

## Table of Contents

- [Run 1: K-Nearest Neighbours](#run-1-tiny-image-knn-classifier)
- [Run 2: Linear Classification with BoVW](#run-2-linear-classification-with-bovw)
- [Run 3: VS-CNN with AlexNet + PCA](#run-3-vs-cnn-with-alexnet--pca)
- [Team & Contribution](#team--contribution)

---

## Run 1: Tiny Image KNN Classifier

**Script**: `run1/scene_classification_tiny_knn.py`  

**Description**:  
This method performs scene classification using a **tiny-image feature representation** combined with a **K-Nearest Neighbors (KNN)** classifier. The process includes training on labeled images, selecting the best `K` via cross-validation, and evaluating predictions on a test set.

### Input Requirements:

Starting at line 191 the following parameters are defined and can be used to direct the code to the correct training and test data paths:

```python
# === Parameters and paths ===
train_dir = "..."
test_dir  = "..."
answers_file = "answer.txt"
pred_file    = "run1.txt"
```

_Assign as necessary by user_

## Run 2: Linear Classification with BoVW

**Script**: `run2/main.py`  

**Description**:  
This method improves upon the KNN approach using Bag-of-Visual-Words (BoVW). It employs local image patches and a clustering method to create a visual vocabulary. A linear classifier is then used to predict scene categories.

### Input Requirements:

At line 365 and 366 the following folder are defined:

```python
# Define the folder paths for training and testing data
folder_path = "training/"
test_folder = "testing/"
```

_Assign as necessary by user_ - It will produce a warning if the folders provided are not available.

## Run 3: VS-CNN with AlexNet + PCA

**Scripts**:  
- `image_preprocessing.py`: Prepares images for feature extraction
- `vsr_generator.py`: Generates saliency maps
- `vsre_generator.py`: Creates enhanced images using saliency maps
- `pre_feat_ext.py`: Orchestrates the preprocessing pipeline
- `full_feature_extraction.py`: Extracts deep features using AlexNet
- `pca_classifier_validate.py`: Cross-validates model performance
- `pca_classifier_train.py`: Trains the final model
- `pca_classifier_predict.py`: Generates predictions on test data
- `denseSIFT.py`: Traditional baseline for comparison

**Description**:  
Our custom approach is a modified version of the Visually Sensitive CNN (VS-CNN) that combines deep feature extraction with saliency-based region enhancement. The pipeline consists of three main stages:

1. **Image Preprocessing & Enhancement**:
   - Convert grayscale images to RGB and resize to 227×227 pixels
   - Generate saliency maps using a patch-based approach
   - Create enhanced images by multiplying the original with saliency maps

2. **Deep Feature Extraction**:
   - Use a pre-trained AlexNet (Places365 weights) to extract fc6 and fc7 features
   - Extract features from three image variants: original, VSR (saliency map), and VSRE (enhanced)
   - Concatenate all features into a 24,576-dimensional vector (6 features × 4,096)

3. **Dimensionality Reduction & Classification**:
   - Apply PCA to reduce dimensions while preserving 99% of variance
   - Use a linear SVM classifier on the PCA-reduced features
   - Validate using 5-fold cross-validation (86.87% mean accuracy)

### Input Requirements:

To run the complete VS-CNN pipeline, follow these steps:

1. **Setup file paths in each script**:
   - Update paths in `pre_feat_ext.py` to point to your training data
   - Configure paths in `full_feature_extraction.py` for the AlexNet model

2. **Run preprocessing and feature extraction**:
   ```bash
   python pre_feat_ext.py
   python full_feature_extraction.py
   ```
   This will generate `vs_cnn_train_features.h5` containing features for all training images.

3. **Validate, train, and predict**:
   ```bash
   python pca_classifier_validate.py  # Outputs cross-validation results
   python pca_classifier_train.py     # Creates svm_pca_model.joblib
   python pca_classifier_predict.py   # Generates run3.txt with predictions
   ```

Note: The AlexNet weights file (`alexnet_places365.pth`) should be in the working directory.
