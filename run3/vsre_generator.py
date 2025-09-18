# This script creates the enhanced salency maps (visually sensitive region enhanced)

import os
import cv2
import numpy as np
from pathlib import Path

def create_enhanced_images(original_folder, saliency_folder, output_folder):
    """
    Create visually sensitive region enhancement (VSRE) images by multiplying
    original images with their corresponding saliency maps.
    
    This implements the enhancement described in the paper as:
    fe(i,j) = fs(i,j) * f(i,j)
    Where:
    - fe is the enhanced image
    - fs is the saliency map
    - f is the original image
    
    Args:
        original_folder: Folder containing original processed RGB images
        saliency_folder: Folder containing saliency maps (vsr)
        output_folder: Folder to save enhanced images
        
    Raises:
        FileNotFoundError: If original_folder doesn't exist
        FileNotFoundError: If saliency_folder doesn't exist
        FileNotFoundError: If specific image or saliency map can't be found
    """
    # Check if input folders exist
    if not os.path.exists(original_folder):
        raise FileNotFoundError(f"Original images folder not found: {original_folder}")
    
    if not os.path.exists(saliency_folder):
        raise FileNotFoundError(f"Saliency maps folder not found: {saliency_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Get all image files in the original folder
    original_files = [f for f in os.listdir(original_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(original_files)} original images to process")
    
    for i, original_file in enumerate(original_files):
        print(f"Processing image {i+1}/{len(original_files)}: {original_file}")
        
        # Construct path to original image
        original_path = os.path.join(original_folder, original_file)
        
        # Construct expected saliency filename
        base_name = os.path.splitext(original_file)[0]
        saliency_file = f"{base_name}.jpg"  
        saliency_path = os.path.join(saliency_folder, saliency_file)
        
        # Check if saliency map exists
        if not os.path.exists(saliency_path):
            raise FileNotFoundError(f"Saliency map not found for {original_file}: {saliency_path}")
        
        # Construct output filename and path (with _enhanced suffix)
        output_file = f"{base_name}_enhanced.jpg"
        output_path = os.path.join(output_folder, output_file)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"  Enhanced image already exists, skipping")
            continue
        
        # Read original image and saliency map
        original_img = cv2.imread(original_path)
        if original_img is None:
            raise FileNotFoundError(f"Error loading original image: {original_path}")
            
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
        if saliency_map is None:
            raise FileNotFoundError(f"Error loading saliency map: {saliency_path}")
        
        # Ensure saliency map has same dimensions as original image
        if original_img.shape[:2] != saliency_map.shape:
            print(f"  Resizing saliency map to match original image dimensions")
            saliency_map = cv2.resize(saliency_map, (original_img.shape[1], original_img.shape[0]))
        
        # Normalize saliency map to [0, 1]
        saliency_map_norm = saliency_map.astype(float) / 255.0
        
        # Create enhanced image: element-wise multiplication as per paper formula
        # fe(i,j) = fs(i,j) * f(i,j)
        # Need to expand saliency map from 2D to 3D to match color channels
        saliency_map_3d = np.expand_dims(saliency_map_norm, axis=2)
        saliency_map_3d = np.repeat(saliency_map_3d, 3, axis=2)
        
        # Apply enhancement by multiplication
        enhanced_img = (original_img.astype(float) * saliency_map_3d).astype(np.uint8)
        
        # Save enhanced image
        cv2.imwrite(output_path, enhanced_img)
        print(f"  Saved enhanced image to {output_path}")
    
    print("All enhanced images created")


if __name__ == "__main__":
    # Use your specific paths - update these for your environment
    original_folder = r"path_to_output\\processed\\bedroom"
    saliency_folder = r"path_to_output\\processed\\bedroom_vsr"
    output_folder = r"path_to_output\\processed\\bedroom_vsre"
    
    # Run the enhancement function
    create_enhanced_images(original_folder, saliency_folder, output_folder)