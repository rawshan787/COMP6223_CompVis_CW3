# This script does the image preparation to ensure compatability with AlexNet model

import os
import cv2
import numpy as np
from pathlib import Path

def process_scene15_images(input_folder, output_folder):
    """
    Process images from Scene-15 dataset:
    1. Check if grayscale and convert to RGB if needed
    2. Resize to 227x227 pixels (AlexNet input size)
    3. Save in RGB format (required for AlexNet)
    
    Args:
        input_folder: Path to folder containing Scene-15 images
        output_folder: Path to save processed images
        
    Raises:
        FileNotFoundError: If input_folder doesn't exist
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, img_file in enumerate(image_files):
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)
        
        # Print progress for every 10 images
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
            
        # Read image using OpenCV
        img = cv2.imread(input_path)
        
        # Check if image was loaded properly
        if img is None:
            raise FileNotFoundError(f"Error loading image: {input_path}")
            
        # Check if image is grayscale (by counting channels)
        if len(img.shape) == 2 or img.shape[2] == 1:
            # Image is grayscale - convert to RGB by duplicating channel
            print(f"Converting {img_file} from grayscale to RGB")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # Image is already in BGR format from OpenCV - convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 227x227 (standard AlexNet input size)
        img_resized = cv2.resize(img_rgb, (227, 227), interpolation=cv2.INTER_LANCZOS4)
        
        # Save the processed image in RGB format
        # Since OpenCV's imwrite expects BGR, we need to convert back for saving
        cv2.imwrite(output_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    print(f"All images processed and saved to {output_folder}")


if __name__ == "__main__":
    # Define specific paths - these paths should be updated for the environment
    input_folder = r"path_to_data\\training\\bedroom"
    output_folder = r"path_to_output\\processed\\bedroom"
    
    # Run the processing function
    process_scene15_images(input_folder, output_folder)