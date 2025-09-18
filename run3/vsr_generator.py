# This script creates the salency maps (visually sensitive regions)

import os
import cv2
import numpy as np
from pathlib import Path
from skimage import color
from scipy.spatial.distance import cdist  # For fast distance calculation
from scipy.interpolate import griddata



def compute_optimized_saliency(img, num_scales=2, patch_size=7, M=20, c=3.0):
    """
    Compute visual saliency map for an image using patch-based approach.
    
    This function implements a patch-based saliency detection algorithm that:
    1. Converts image to LAB color space (perceptually uniform)
    2. Extracts patches from the image
    3. Computes patch-to-patch distances based on color and position
    4. Determines saliency values based on distance metrics
    
    Args:
        img: Input image (BGR format from OpenCV)
        num_scales: Number of scales to use in multi-scale approach
        patch_size: Size of square patches to extract
        M: Number of most similar patches to consider
        c: Position influence factor
        
    Returns:
        saliency_map: Normalized saliency map (values 0-1)
    """
    # Original dimensions
    height, width = img.shape[:2]
    
    # Convert to LAB color space (perceptually uniform)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Downsample for faster processing (optional)
    downsample_factor = 1
    lab_img = cv2.resize(lab_img, (int(width * downsample_factor), int(height * downsample_factor)))
    height, width = lab_img.shape[:2]
    
    # Initialize multi-scale saliency map
    saliency_map = np.zeros((height, width), dtype=np.float32)
    
    # Extract patches with stride for efficiency
    stride = max(1, patch_size // 2)  # Stride to reduce patch count
    patches = []
    positions = []
    
    # Loop through image to extract patches
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if y + patch_size <= height and x + patch_size <= width:
                # Extract patch and flatten to vector
                patch = lab_img[y:y+patch_size, x:x+patch_size].flatten()
                patches.append(patch)
                positions.append((y, x))
    
    # Convert lists to numpy arrays for vectorized operations
    patches = np.array(patches)
    positions = np.array(positions)
    
    print(f"  Computing patch distances for {len(patches)} patches...")
    
    # Calculate color-based distances between all patches at once using cdist
    color_dists = cdist(patches, patches, 'euclidean')
    
    # Calculate position-based distances
    pos_dists = cdist(positions, positions, 'euclidean')
    
    # Combine color and position distances with position influence factor c
    combined_dists = color_dists / (1.0 + c * pos_dists)
    
    # Set self-distances to infinity to ignore them
    np.fill_diagonal(combined_dists, np.inf)
    
    print("  Computing saliency values...")
    
    # For each patch, find M most similar patches and compute saliency
    for i in range(len(patches)):
        # Get M most similar patches (lowest distance)
        most_similar_idx = np.argsort(combined_dists[i])[:M]
        most_similar_dists = combined_dists[i, most_similar_idx]
        
        # Compute saliency value (1 - similarity)
        saliency_value = 1.0 - np.exp(-1.0/M * np.sum(most_similar_dists))
        
        # Update saliency map at patch position
        y, x = positions[i]
        saliency_map[y:y+patch_size, x:x+patch_size] = saliency_value
    
    # Interpolate any holes in saliency map
    mask = saliency_map > 0
    if not np.all(mask):
        y_coords, x_coords = np.indices(saliency_map.shape)
        points = np.column_stack((y_coords[mask].ravel(), x_coords[mask].ravel()))
        values = saliency_map[mask].ravel()
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        saliency_map = griddata(points, values, (grid_y, grid_x), method='nearest')
    
    # Resize back to original size if downsampled
    if downsample_factor != 1.0:
        saliency_map = cv2.resize(saliency_map, (int(width / downsample_factor), int(height / downsample_factor)))
    
    # Normalize saliency map to [0, 1]
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply simple Gaussian smoothing to reduce noise
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    
    return saliency_map



def process_images_for_saliency_optimized(input_folder, output_folder):
    """
    Process all images in input folder to generate visual saliency region (VSR) maps.
    
    For each image, this function:
    1. Computes saliency using patch-based approach
    2. Saves grayscale saliency maps to output folder
    
    Args:
        input_folder: Path to folder containing preprocessed images
        output_folder: Path to save saliency maps
        
    Raises:
        FileNotFoundError: If input_folder doesn't exist
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        
        input_path = os.path.join(input_folder, img_file)
        output_filename = os.path.splitext(img_file)[0] + '.jpg'
        output_path = os.path.join(output_folder, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"  Saliency map already exists, skipping")
            continue
        
        # Read image
        img = cv2.imread(input_path)
        
        # Check if image was loaded properly
        if img is None:
            raise FileNotFoundError(f"Error loading image: {input_path}")
        
        # Compute saliency
        saliency_map = compute_optimized_saliency(img)
        
        # Save saliency map
        saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)
        cv2.imwrite(output_path, saliency_map_uint8)
        print(f"  Saved saliency map to {output_path}")
    
    print("All saliency maps created")

if __name__ == "__main__":
    # Use your paths - update these for your environment
    input_folder = r"path_to_output\\processed\\bedroom"
    output_folder = r"path_to_output\\processed\\bedroom_vsr"

    # Run the optimized function
    process_images_for_saliency_optimized(input_folder, output_folder)