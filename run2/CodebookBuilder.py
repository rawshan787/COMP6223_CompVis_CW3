import numpy as np
from sklearn.cluster import KMeans

class CodebookBuilder:
    def __init__(self, num_clusters, sample_size, random_state):
        """
        Initialize the CodebookBuilder with clustering parameters.

        Parameters:
        num_clusters (int): Number of visual words (clusters) to form in the codebook.
        sample_size (int): Maximum number of image patches to use when fitting KMeans.
        random_state (int): Seed for the random number generator to ensure reproducibility.

        Initializes an empty KMeans model that will be trained later using sampled patches.
        """
        
        # Initialize the CodebookBuilder with key parameters for clustering
        self.num_clusters = num_clusters # Number of visual words (clusters) to form during KMeans clustering
        self.sample_size = sample_size # Maximum number of patches to use for the KMeans clustering
        self.random_state = random_state # Random seed for reproducibility of the results
        self.kmeans = None # Placeholder for the KMeans model after training

    def build_codebook(self,patches):
        """
        Build the codebook by clustering patches using KMeans.
        
        Arguments:
        patches -- Array or list of image patches to be clustered
        
        Returns:
        cluster_centers_ -- Centers of the clusters (visual vocabulary)
        """
        
        # Check if the number of patches exceeds the sample size. If so, subsample the patches
        if len(patches) > self.sample_size:
            # Randomly select a subset of patches up to the sample size
            indices = np.random.choice(len(patches), self.sample_size, replace=False)
            sampled_patches = patches[indices]
        else:
            # Use all patches if the number is less than or equal to the sample size
            sampled_patches = patches

        # Log the number of patches being used for clustering
        print(f"Running KMeans on {len(sampled_patches)} patches with {self.num_clusters} clusters")

        # Initialize the KMeans model with the specified number of clusters and random state
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state, n_init='auto')

        # Fit the KMeans model to the sampled patches to form clusters
        self.kmeans.fit(sampled_patches)

        # Return the cluster centers which represent the visual vocabulary (codebook)
        return self.kmeans.cluster_centers_
        
    def get_kmeans(self):
        """
        Retrieve the trained KMeans model.
        
        Returns:
        kmeans -- The trained KMeans model object
        """
         
        return self.kmeans # Return the trained KMeans model

