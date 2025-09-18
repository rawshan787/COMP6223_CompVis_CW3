import os
import time
import numpy as np
import joblib
import sys
from glob import glob

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

from PatchExtractor import PatchExtractor
from CodebookBuilder import CodebookBuilder
from BOVWHistogram import BOVWHistogram

class BoVWSceneClassifier:
    """
    Bag-of-Visual-Words Scene Classifier:
    This class encapsulates the process of extracting features using patches,
    clustering them to form a codebook, computing histograms for images, and
    training an image classifier using a linear SVM.
    """

    def __init__(self, patch_size=8, stride=4, num_clusters=500, sample_size=100000, use_presaved=False):
        """
        Constructor for the BoVWSceneClassifier class.

        Initializes the key components required for the Bag-of-Visual-Words (BoVW) based 
        scene classification pipeline. These components include patch extraction, codebook 
        generation via clustering, and setup for classifier training or loading.

        Parameters:
            patch_size (int): Size (in pixels) of each square patch to extract from input images.
            stride (int): Step size (in pixels) for moving the patch extraction window across the image.
            num_clusters (int): Number of visual words (i.e., clusters) for the visual vocabulary (codebook).
            sample_size (int): Maximum number of patches sampled from the dataset for clustering.
            use_presaved (bool): If True, load pre-trained codebook and classifier models 
                                instead of generating/training from scratch.
        """

        # Store configuration parameters
        self.patch_size = patch_size
        self.stride = stride
        self.num_clusters = num_clusters
        self.sample_size = sample_size
        self.use_presaved = use_presaved

        # Initialize patch extractor with given patch size and stride
        self.extractor = PatchExtractor(patch_size, stride)
        # Initialize codebook builder (typically wraps KMeans or similar clustering method)
        self.builder = CodebookBuilder(num_clusters, sample_size, random_state=42)

        # Placeholder for histogram handler (will process patch-to-visual-word histograms)
        self.histHandler = None
        # Placeholder for the trained classifier (e.g., SVM, RandomForest)
        self.classifier = None
        # Mapping from class labels (e.g., integers) to human-readable names
        self.label_map = {}


    def extract_features_and_labels(self, folder_path):
        """
        Extracts Bag-of-Visual-Words (BoVW) feature histograms and corresponding labels
        from a directory structure of images organized by class folders.

        Each image is processed to generate a normalized histogram of visual word frequencies,
        representing its appearance based on the learned visual vocabulary.

        Parameters:
            folder_path (str): Path to the root folder containing subdirectories of images.
                            Each subdirectory represents a separate class.

        Returns:
            X (np.ndarray): Array of BoVW histograms (features) for all images.
            y (np.ndarray): Array of corresponding numeric class labels.
        """

        print("[extract] Extracting histograms and labels from training images...") 

        # Get sorted list of class subfolders and build a label mapping
        class_folders = sorted(os.listdir(folder_path))
        self.label_map = {cls: idx for idx, cls in enumerate(class_folders)}

        # Lists to hold feature vectors and labels
        X, y = [], []

        # Iterate through each class folder
        for cls in class_folders:
            print(f"[extract] Processing class '{cls}'...")
            class_path = os.path.join(folder_path, cls)

            # Process all JPEG images in the class folder
            for img_path in glob(os.path.join(class_path, '*.jpg')):

                # Extract patches from the image using the patch extractor
                patches = self.extractor.extract_image_patch_from_path(img_path)
                
                # Convert the set of patches into a visual word histogram
                hist = self.histHandler.extract_histogram(patches)

                # Only add valid (non-empty) histograms
                if hist.size > 0:
                    X.append(hist)
                    y.append(self.label_map[cls])

        print(f"[extract] Completed with {len(X)} samples.")
        
        # Return features and labels as NumPy arrays
        return np.array(X), np.array(y)
    

    def train(self, folder_path):
        """
        Executes the full training pipeline for the BoVWSceneClassifier.

        The pipeline includes:
        1. Patch extraction from training images.
        2. Codebook (visual vocabulary) construction using KMeans clustering.
        3. Conversion of images into BoVW histograms based on the codebook.
        4. Training a linear One-vs-Rest classifier on the extracted features.
        5. Saving the trained models and label mappings to disk.

        Parameters:
            folder_path (str): Path to the root directory containing class-labeled
                            subdirectories of training images.
        """

        print("[train] Extracting patches...")

        # Step 1: Extract patches from all training images for codebook generation
        patches = self.extractor.extract_patches_from_folder(folder_path)

        print("[train] Building codebook...")

        # Step 2: Build the visual vocabulary using KMeans on extracted patches
        self.builder.build_codebook(patches)

        # Step 3: Initialize histogram extractor with the trained KMeans model
        self.histHandler = BOVWHistogram(self.builder.get_kmeans(), self.patch_size, self.stride)

        # Step 4: Convert images to BoVW histograms and extract corresponding labels
        X, y = self.extract_features_and_labels(folder_path)

        print("[train] Training classifier...")

        # Step 5: Train a One-vs-Rest linear SVM classifier on the BoVW histograms
        self.classifier = OneVsRestClassifier(LinearSVC())
        self.classifier.fit(X, y)
        
        # Step 6: Save the trained classifier, KMeans model, and label map to disk
        joblib.dump(self.classifier, "bovw_classifier.joblib")
        joblib.dump(self.builder.get_kmeans(), "kmeans_model.joblib")
        joblib.dump(self.label_map, "label_map.joblib")
        print("[train] Model saved.")


    def load(self):
        """
        Loads previously saved classifier model, KMeans codebook, and label mappings.

        This method is used when `use_presaved=True` to bypass the training process
        and utilize an existing trained pipeline. It restores the classifier and
        visual vocabulary, then reinitializes the histogram handler accordingly.
        """

        # Load the trained One-vs-Rest classifier from disk
        self.classifier = joblib.load("bovw_classifier.joblib")

        # Load the pre-trained KMeans model used for visual word assignment
        kmeans_model = joblib.load("kmeans_model.joblib")

        # Load the class label-to-index mapping
        self.label_map = joblib.load("label_map.joblib")

        # Reinitialize the histogram handler with the loaded KMeans model
        self.histHandler = BOVWHistogram(kmeans_model, self.patch_size, self.stride)

        print("[load] Loaded pretrained models and mappings.")


    def cross_validate(self, folder_path, param_grid, report=False):
        """
        Performs cross-validation to tune hyperparameters and find the best configuration for the classifier.

        The method iterates over a grid of hyperparameters (patch size, stride, and number of clusters) and 
        evaluates the performance of the model using 5-fold cross-validation. The best performing configuration 
        based on mean accuracy is then selected. Optionally, a classification report can be generated for detailed 
        performance evaluation on the validation sets.

        Parameters:
            folder_path (str): Path to the folder containing class-labeled subdirectories of images for training.
            param_grid (dict): A dictionary containing the hyperparameter search space. The keys are 'patch_sizes', 
                            'strides', and 'num_clusters', with each mapping to a list of possible values.
            report (bool): If True, generates a classification report with detailed per-class performance.
        
        Returns:
            None: The method prints the best hyperparameters and the mean accuracy, and retrains the final model 
                using the optimal configuration.
        """

        print("[cv] === Starting Cross-Validation ===")

        # Initialize best score and parameter tracking
        best_score = 0
        best_params = {}
        all_results = {}  # Store predictions from the best model if report = True

        # Stratified 5-fold cross-validation setup
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search over the parameter space
        for patch_size in param_grid['patch_sizes']:
            for stride in param_grid['strides']:
                for num_clusters in param_grid['num_clusters']:
                    print(f"[cv] Testing patch={patch_size}, stride={stride}, clusters={num_clusters}")

                    # Create a new patch extractor and codebook builder for each parameter configuration
                    extractor = PatchExtractor(patch_size, stride)
                    builder = CodebookBuilder(num_clusters, self.sample_size, random_state=42)

                    # Extract patches and build codebook for the current parameter combination
                    patches = extractor.extract_patches_from_folder(folder_path)
                    builder.build_codebook(patches)

                    # Create histogram handler for the current codebook
                    hist_handler = BOVWHistogram(builder.get_kmeans(), patch_size, stride)

                    # Set the current extractor and histogram handler
                    self.extractor = extractor
                    self.histHandler = hist_handler

                    # Generate label map for the current set of images
                    self.label_map = {cls: idx for idx, cls in enumerate(sorted(os.listdir(folder_path)))}

                    # Extract features and labels from the dataset
                    X, y = self.extract_features_and_labels(folder_path)

                    # Skip the fold if there are less than 2 classes in the validation set
                    if len(np.unique(y)) < 2:
                        continue  # Skip invalid folds

                    # Perform 5-fold cross-validation on the current configuration
                    scores = []
                    y_true_all = []
                    y_pred_all = []
                    for train_idx, val_idx in kfold.split(X, y):
                        clf = OneVsRestClassifier(LinearSVC())
                        clf.fit(X[train_idx], y[train_idx])
                        y_pred = clf.predict(X[val_idx])
                        score = accuracy_score(y[val_idx], y_pred)
                        scores.append(score)

                        # If a report is requested, store true and predicted values
                        if report:
                            y_true_all.extend(y[val_idx])
                            y_pred_all.extend(y_pred)
            
                    # Compute the mean accuracy for this parameter combination
                    mean_score = np.mean(scores)
                    print(f"    Mean CV accuracy: {mean_score:.4f}")

                    # Track the best score and parameters
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'patch_size': patch_size,
                            'stride': stride,
                            'num_clusters': num_clusters
                        }
                        
                        # If report is requested, store classification results
                        if report:
                            all_results['y_true'] = y_true_all
                            all_results['y_pred'] = y_pred_all

        # Report the best parameters and accuracy found
        print(f"[cv] Best parameters: {best_params} with accuracy {best_score:.4f}")

        # If report is enabled, print the detailed classification report for the best model
        if report and 'y_true' in all_results:
            print("[cv] Classification report for best model:")
            print(classification_report(all_results['y_true'], all_results['y_pred'], target_names=sorted(self.label_map)))

            # Confusion matrix
            cm = confusion_matrix(all_results['y_true'], all_results['y_pred'])
            print("[cv] Confusion matrix for best model:")
            print(cm)

        # Use the best parameters to retrain the final model on the entire dataset
        self.patch_size = best_params['patch_size']
        self.stride = best_params['stride']
        self.num_clusters = best_params['num_clusters']
        self.extractor = PatchExtractor(self.patch_size, self.stride)
        self.builder = CodebookBuilder(self.num_clusters, self.sample_size, random_state=42)
        self.train(folder_path)

    def predict_folder(self, test_folder, output_file="run2.txt"):
        """
        Predicts the class labels for all images in the given test folder and writes the results to an output file.

        The method processes each image, extracts features using the previously trained feature extractor and 
        histogram handler, classifies the features using the trained classifier, and saves the results to a text file.

        Parameters:
            test_folder (str): Path to the folder containing test images.
            output_file (str): The name of the file where the predictions will be saved. Default is "run2.txt".
        
        Returns:
            None: Saves the predicted class labels for each image in the test folder to the specified output file.
        """
        print("[predict] Starting prediction...")

        # Get all image paths in the test folder
        image_paths = glob(os.path.join(test_folder, '*.jpg'))
        X_test = []

        # Extract patches and compute histograms for all test images
        for img_path in image_paths:
            patches = self.extractor.extract_image_patch_from_path(img_path)
            hist = self.histHandler.extract_histogram(patches)
            X_test.append(hist)

        X_test = np.array(X_test)

        # Raise an error if no valid test samples are found
        if X_test.shape[0] == 0:
            raise ValueError("No valid test samples found. Cannot run prediction.")

        # Predict the class labels for all test images
        y_pred = self.classifier.predict(X_test)

        # Create a reverse map from numeric class labels to class names
        reverse_map = {v: k for k, v in self.label_map.items()}

        # Helper function to extract numeric values from image filenames for sorting
        def extract_number(filename):
            basename = os.path.basename(filename)
            digits = ''.join(filter(str.isdigit, basename))
            return int(digits) if digits else 0

        # Sort image paths based on the numeric values extracted from filenames
        sorted_paths = sorted(image_paths, key=extract_number)

        # Prepare output lines in the format: image_name class_label
        output_lines = []
        for img_path in sorted_paths:
            idx = image_paths.index(img_path)
            pred_label = y_pred[idx]
            pred_class = reverse_map[pred_label]
            output_lines.append(f"{os.path.basename(img_path)} {pred_class}")

        # Save the predictions to the specified output file
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))

        print(f"[predict] Predictions saved to {output_file}")


if __name__ == "__main__":

    # Start the timer to measure execution time
    start = time.time()

    # Define the folder paths for training and testing data
    folder_path = "training/"
    test_folder = "testing/"

    # Check if the folders exist, and exit with a warning if they don't
    if not os.path.exists(folder_path):
        print(f"Warning: The training folder '{folder_path}' does not exist!")
        sys.exit(1)  # Exit the program with a non-zero status indicating an error

    if not os.path.exists(test_folder):
        print(f"Warning: The testing folder '{test_folder}' does not exist!")
        sys.exit(1)  # Exit the program with a non-zero status indicating an error

    # Initialize the BoVWSceneClassifier model with the specified parameters
    model = BoVWSceneClassifier(patch_size=4, stride =2, num_clusters= 800, use_presaved=False)

    # Load a pre-trained model if 'use_presaved' is set to True
    if model.use_presaved:
        model.load()
    else:
        # If the model isn't using pre-saved parameters, train it from scratch
        model.train(folder_path)

        # Uncomment and specify parameter grid for cross-validation if necessary
        '''param_grid = {
        'patch_sizes': [4],
        'strides': [2],
        'num_clusters': [800]
        }

        #Perform Cross Validation using above parameter grid
        model.cross_validate(folder_path, param_grid, report=True)'''

    # Predict the class labels for all test images and output the results to a file
    model.predict_folder(test_folder)

    # Print the total time taken for the execution of the entire process
    print(f"Done in {time.time() - start:.2f} seconds.")
