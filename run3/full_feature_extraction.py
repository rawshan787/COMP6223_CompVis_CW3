# This script does the full feature extraction by applying the fully connected layers from AlexNet model

import os
import cv2
import numpy as np
import h5py
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# === SCENE LABELS ===
# List of scene categories used in the dataset
scene_labels = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'Insidecity',
                'Kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry',
                'store', 'Street', 'Suburb', 'TallBuilding']
# Create a mapping from scene names to numeric indices
label_map = {name: idx for idx, name in enumerate(scene_labels)}


# === MODEL LOADING ===
def load_model(weights_path):
    """
    Load a pre-trained AlexNet model and modify it for feature extraction.
    
    Args:
        weights_path: Path to the pre-trained weights file
        
    Returns:
        A feature extraction model that outputs fc6 and fc7 features
    """
    # Initialize AlexNet architecture without pre-trained weights
    model = models.alexnet(weights=None)
    # Modify the final layer to match the Places365 dataset (365 scene categories)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 365)
    
    # Load pre-trained weights
    state = torch.load(weights_path, map_location=torch.device('cpu'))
    # Handle state_dict format if present
    if 'state_dict' in state:
        # Remove 'module.' prefix which is added when trained with DataParallel
        state = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
    model.load_state_dict(state)
    model.eval()  # Set model to evaluation mode

    # Create a custom feature extractor module
    class FeatureExtractor(nn.Module):
        """
        Wrapper around AlexNet to extract fc6 and fc7 features
        """
        def __init__(self, base):
            super().__init__()
            self.features = base.features      # Convolutional layers
            self.avgpool = base.avgpool        # Average pooling layer
            self.classifier = base.classifier  # Fully connected layers
            
        def forward(self, x):
            # Process through convolutional layers
            x = self.features(x)
            # Apply average pooling
            x = self.avgpool(x)
            # Flatten the tensor for the fully connected layers
            x = torch.flatten(x, 1)
            
            # Process through first 5 classifier layers (up to fc6)
            for i in range(5):
                x = self.classifier[i](x)
            # Save fc6 features
            fc6 = x.clone()
            
            # Apply ReLU and dropout to get fc7 input
            x = self.classifier[5](x)
            # Save fc7 features
            fc7 = x.clone()
            
            return {'fc6': fc6, 'fc7': fc7}
            
    return FeatureExtractor(model)


# === IMAGE PREPROCESSING PIPELINE ===
# Standard image transformations for AlexNet
transform = transforms.Compose([
    transforms.ToPILImage(),         # Convert to PIL Image
    transforms.Resize(256),          # Resize to 256x256
    transforms.CenterCrop(224),      # Center crop to 224x224
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(            # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# === FEATURE EXTRACTION FROM SINGLE IMAGE ===
def extract_fc6_fc7(model, img_path):
    """
    Extract fc6 and fc7 features from a single image
    
    Args:
        model: The feature extraction model
        img_path: Path to the image file
        
    Returns:
        Tuple of (fc6, fc7) feature vectors
    """
    # Read image with OpenCV
    img = cv2.imread(img_path)
    
    # Return zeros if image loading failed
    if img is None:
        return np.zeros(4096), np.zeros(4096)
        
    # Convert from BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Extract features without computing gradients
    with torch.no_grad():
        out = model(img_tensor)
        
    # Return numpy arrays of features
    return out['fc6'].squeeze(0).numpy(), out['fc7'].squeeze(0).numpy()


# === MAIN FEATURE EXTRACTION FUNCTION ===
def extract_all_features(ori_root, vsr_root, vsre_root, weights_path, output_file):
    """
    Extract features from all images in the dataset
    
    Args:
        ori_root: Path to original images
        vsr_root: Path to VSR images
        vsre_root: Path to VSRE images
        weights_path: Path to model weights
        output_file: Path for saving features HDF5 file
    """
    # Load the feature extraction model
    print("Loading model...")
    model = load_model(weights_path)
    
    # Initialize lists to store data
    all_features = []  # Feature vectors
    all_labels = []    # Class labels
    all_filenames = []  # Image filenames
    
    print(f"Starting feature extraction for {len(scene_labels)} scene categories")

    # Process each scene category
    for scene in scene_labels:
        # Construct paths to scene folders in each dataset variant
        ori_folder = os.path.join(ori_root, scene)
        vsr_folder = os.path.join(vsr_root, scene)
        vsre_folder = os.path.join(vsre_root, scene)
        
        # Get all JPEG files in the folder
        files = [f for f in os.listdir(ori_folder) if f.lower().endswith('.jpg')]
        
        # Process each image in the current scene category
        for fname in tqdm(files, desc=f"Processing {scene}"):
            # Get base filename without extension
            base = os.path.splitext(fname)[0]
            
            # Construct full image paths for original, VSR, and VSRE versions
            ori_path = os.path.join(ori_folder, fname)
            vsr_path = os.path.join(vsr_folder, fname)
            vsre_path = os.path.join(vsre_folder, fname)
            
            # Extract features from original image
            fc6_o, fc7_o = extract_fc6_fc7(model, ori_path)
            # Extract features from VSR image
            fc6_s, fc7_s = extract_fc6_fc7(model, vsr_path)
            # Extract features from VSRE image
            fc6_e, fc7_e = extract_fc6_fc7(model, vsre_path)
            
            # Concatenate all features into a single vector
            # Order: fc7_original, fc6_original, fc7_vsr, fc6_vsr, fc7_vsre, fc6_vsre
            # Shape: (24576,) = 6 * 4096
            fc_vector = np.concatenate([fc7_o, fc6_o, fc7_s, fc6_s, fc7_e, fc6_e])
            
            # Store features, label, and filename
            all_features.append(fc_vector)
            all_labels.append(label_map[scene])
            all_filenames.append(f"{scene}/{fname}")
    
    print(f"Extracting completed. Saving features for {len(all_features)} images...")
    
    # Save extracted features to HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create datasets for features, labels, and filenames
        f.create_dataset('features', data=np.array(all_features, dtype=np.float32))
        f.create_dataset('labels', data=np.array(all_labels, dtype=np.int32))
        f.create_dataset('filenames', data=np.array(all_filenames, dtype=h5py.string_dtype()))

    print(f"\nSuccessfully saved features for {len(all_features)} images to {output_file}")


# === SCRIPT ENTRY POINT ===
if __name__ == "__main__":
    # === PATH CONFIGURATION ===
    ori_root = r"path_to_output\\processed"
    vsr_root = r"path_to_output\\vsr"
    vsre_root = r"path_to_output\\vsre"
    weights_path = r"alexnet_places365.pth"
    output_file = r"vs_cnn_train_features.h5"

    # Run the feature extraction process
    extract_all_features(ori_root, vsr_root, vsre_root, weights_path, output_file)
