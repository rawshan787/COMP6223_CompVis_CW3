# This file does the entire pre-processing pipeline, prior to full feature extraction.

import os
from pathlib import Path
from image_preprocessing import process_scene15_images
from vsr_generator import process_images_for_saliency_optimized
from vsre_generator import create_enhanced_images

def ensure_folder_exists(folder_path):
    """
    Check if a folder exists and create it if it doesn't.
    
    Args:
        folder_path: Path to the folder to check/create
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

def process_category(category, base_input, base_processed, base_vsr, base_vsre):
    """
    Process a single category of images through the entire pipeline:
    1. Preprocess images (resize, convert to RGB)
    2. Generate saliency maps
    3. Create enhanced images
    
    Args:
        category: Name of the category folder
        base_input: Root folder containing original images
        base_processed: Root folder for preprocessed images
        base_vsr: Root folder for saliency maps
        base_vsre: Root folder for enhanced images
        
    Raises:
        FileNotFoundError: If input category folder doesn't exist
    """
    # Define per-category folders
    class_input = os.path.join(base_input, category)
    class_processed = os.path.join(base_processed, category)
    class_vsr = os.path.join(base_vsr, category)
    class_vsre = os.path.join(base_vsre, category)
    
    # Check if input category folder exists
    if not os.path.isdir(class_input):
        print(f"Skipping '{category}' - not a directory")
        return
    
    print(f"\n=== Processing category: {category} ===")
    
    # Create output folders if they don't exist
    for folder in [class_processed, class_vsr, class_vsre]:
        ensure_folder_exists(folder)
    
    # Step 1: Preprocess images to RGB + resize
    print(f"Step 1: Preprocessing images for {category}")
    process_scene15_images(class_input, class_processed)
    
    # Step 2: Generate saliency maps
    print(f"Step 2: Generating saliency maps for {category}")
    process_images_for_saliency_optimized(class_processed, class_vsr)
    
    # Step 3: Multiply original × saliency → VSRE images
    print(f"Step 3: Creating enhanced images for {category}")
    create_enhanced_images(class_processed, class_vsr, class_vsre)
    
    print(f"Completed processing for category: {category}")

if __name__ == "__main__":
    # Set root folders - update these for your environment
    base_input = r"path_to_output\\training"
    base_processed = r"path_to_output\\processed"
    base_vsr = r"path_to_output\\vsr"
    base_vsre = r"path_to_output\\vsre"

    # Check if base input folder exists
    if not os.path.exists(base_input):
        raise FileNotFoundError(f"Base input folder not found: {base_input}")
    
    # Create output base folders if they don't exist
    for folder in [base_processed, base_vsr, base_vsre]:
        ensure_folder_exists(folder)
    
    # Loop over each category folder in Scene-15
    categories = [cat for cat in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, cat))]
    print(f"Found {len(categories)} categories to process")
    
    # Process each category
    for category in categories:
        try:
            process_category(category, base_input, base_processed, base_vsr, base_vsre)
        except Exception as e:
            print(f"Error processing category '{category}': {e}")
            # Continue with next category instead of stopping completely
            continue
    
    print("All categories processed")
