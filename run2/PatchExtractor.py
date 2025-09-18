import os
import cv2
import numpy as np

class PatchExtractor:
    """
    A class for extracting image patches from input images with specified patch size and stride.
    The patches are normalized and flattened for further processing, such as machine learning tasks.
    """

    def __init__(self, patch_size, stride):
        """
        Initializes the PatchExtractor with the given patch size and stride.

        Parameters:
        - patch_size (int): The size of the square patches to extract.
        - stride (int): The step size for sliding the window across the image.
        """

        self.patch_size = patch_size # Store the patch size
        self.stride = stride # Store the stride size


    def extract_image_patch(self, image):
        """
        Extracts normalized, flattened patches from the given image using a sliding window approach.

        The function slides a window over the image with the specified stride and extracts
        patches of the specified size. Each patch is normalized by subtracting its mean and
        dividing by its L2 norm, and then it is flattened into a 1D array.

        Parameters:
        - image (numpy.ndarray): The grayscale image from which patches are to be extracted.

        Returns:
        - list: A list of flattened and normalized image patches.
        """

        patches = []    # List to store extracted patches
        h, w = image.shape  # Get the height (h) and width (w) of the image

        # Loop through the image using a sliding window
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):

                # Extract a patch of size patch_size from the image
                patch = image[y:y+self.patch_size, x:x+self.patch_size].astype(np.float32)

                # Normalize the patch: subtract the mean value
                patch -= np.mean(patch)

                # Calculate the L2 norm of the patch
                norm = np.linalg.norm(patch)

                # If the norm is greater than zero, normalize the patch
                if norm > 0:

                    patch /= norm # Normalize by dividing by the L2 norm
                    flat_patch = patch.flatten() # Flatten the patch into a 1D array

                    # Ensure the flattened patch has the correct shape
                    if flat_patch.shape[0] == self.patch_size * self.patch_size:
                        patches.append(flat_patch)  # Add the patch to the list

        return patches # Return the list of patches


    def extract_image_patch_from_path(self, image_path):
        """
        Loads an image from the given path and extracts normalized patches from it.

        Parameters:
        - image_path (str): The file path to the image.

        Returns:
        - list: A list of normalized and flattened patches extracted from the image,
                or an empty list if the image cannot be loaded.
        """

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load the image as grayscale
        if image is None:   # If the image cannot be loaded, return an empty list
            return []
        return self.extract_image_patch(image) # Extract and return the patches

    def get_all_images(self, root_folder):
        """
        Iterates through the directory structure of the root folder and yields grayscale images.

        The method traverses the directory tree and loads all .jpg images from subdirectories.
        It yields each image along with its corresponding class folder name.

        Parameters:
        - root_folder (str): The root directory containing subdirectories of images.

        Yields:
        - tuple: A tuple containing a grayscale image (numpy.ndarray) and the class folder name (str).
        """

        # Iterate through each subdirectory (class folder) in the root folder
        for class_folder in sorted(os.listdir(root_folder)):
            class_path = os.path.join(root_folder, class_folder) # Get the full path to the class folder
            
            # Skip non-directory items
            if not os.path.isdir(class_path):
                continue

            # Iterate through each file in the class folder
            for image_name in os.listdir(class_path):
                # Only process .jpg files
                if image_name.lower().endswith('.jpg'):

                    image_path = os.path.join(class_path, image_name)       # Get the full path to the image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # Load the image as grayscale

                    # If the image was successfully loaded, yield it along with the class folder name
                    if image is not None:
                        yield image, class_folder

    def extract_patches_from_folder(self, root_folder):
        """
        Extracts patches from all images in the specified root folder, iterating through its subdirectories.

        This method utilizes `get_all_images` to retrieve each image and then extracts patches
        from all images in the dataset.

        Parameters:
        - root_folder (str): The root directory containing subdirectories of images.

        Returns:
        - numpy.ndarray: An array of all extracted patches from the images in the dataset.
        """

        all_patches = []    # List to store all extracted patches

        # Iterate through all images in the root folder
        for image, _ in self.get_all_images(root_folder):
            patches = self.extract_image_patch(image)   # Extract patches from the current image
            all_patches.extend(patches) # Add the extracted patches to the list

        # Convert the list of patches to a numpy array and return it
        return np.array(all_patches, dtype=np.float32)
