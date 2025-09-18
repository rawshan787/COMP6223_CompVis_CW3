import numpy as np
from PatchExtractor import PatchExtractor

class BOVWHistogram:
    def __init__(self, kmeans_model, patch_size, stride):
        """
        Initialize the Bag of Visual Words (BoVW) Histogram object.
        
        Arguments:
        kmeans_model -- The trained KMeans model containing the codebook (cluster centers)
        patch_size -- The size of the patches to be extracted from images
        stride -- The step size used for extracting patches from images
        """

        self.kmeans = kmeans_model  # The trained KMeans model, which contains the visual vocabulary (codebook)
        self.num_clusters = self.kmeans.n_clusters # Number of visual words (clusters) in the codebook
        self.extractor = PatchExtractor(patch_size, stride) # PatchExtractor object for extracting image patches

    def extract_histogram(self, patches):
        """
        Extract the histogram of visual word occurrences (BoVW) from the given image patches.
        
        Arguments:
        patches -- Array or list of image patches to be processed
        
        Returns:
        hist -- Normalized histogram representing the distribution of visual words in the patches
        """

        # Normalize patches by mean-centering (subtract mean of each patch)
        patches = patches - np.mean(patches, axis=1, keepdims=True)
        
        # Predict the visual word IDs by assigning each patch to the closest cluster center
        word_ids = self.kmeans.predict(patches)
        
        # Build the histogram by counting occurrences of each visual word
        hist = np.bincount(word_ids, minlength=self.num_clusters).astype(np.float32)

        # Normalize the histogram (L2 normalization)
        hist /= np.linalg.norm(hist) + 1e-6

        return hist # Return the normalized histogram of visual word occurrences