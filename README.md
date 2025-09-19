# Coursework 3: Scene Recognition

![Figure 1: Illustration of the three image types used in VS-CNN:
 original inputs (left), VSR images (middle), and VSRE images
 (right)](features.png)
Figure 1: Illustration of the three image types used in VS-CNN:
 original inputs (left), VSR images (middle), and VSRE images
 (right)



This project explores multiple approaches to scene recognition on the 15-scene dataset, advancing from traditional to deep learning-based methods. We begin with a K-Nearest Neighbours classifier using tiny image features, achieving 22.01% accuracy. Next, a Bag-of-Visual-Words model with a linear classifier improves accuracy to 70% by leveraging local image patches and clustering. Finally, a modified Deep Visually Sensitive CNN (VS-CNN) incorporates context-based saliency detection and AlexNet features, followed by PCA and SVM classification, achieving 86.87% accuracy.

We were awarded First Class (100%) grade for this submission.

## 1. Run 1: K-Nearest-Neighbors

### 1.1. Task Summary
A K-Nearest Neighbors classifier was implemented using tiny image features (16x16 center-cropped, flattened, and normalized vectors) to establish a performance baseline. The optimal hyperparameter (K=7) was determined through 5-fold cross-validation. This configuration achieved 22.01% accuracy on the test set.

### 1.2. Analysis
The tiny image representation proved fundamentally limited by its discard of high-frequency information and structural details. The resulting 22.01% accuracy confirms that while computationally trivial, this approach lacks the discriminative power necessary for meaningful scene recognition, necessitating more sophisticated feature extraction methods.

## 2. Run 2: Linear Classification with BoVW

### 2.1. Task Summary
A Bag-of-Visual-Words model was constructed by extracting image patches via sliding window, clustering with K-Means to form a visual vocabulary, and encoding images as histograms of visual word occurrences. A one-vs-rest linear SVM was trained on these representations. Parameter optimization via grid search yielded an optimal configuration (patch size=4, stride=2, 800 clusters) achieving 70.0% accuracy.

### 2.2. Analysis
Performance exhibited significant class-dependent variance, with high accuracy on scenes containing large distinctive features (e.g., "Forest") and poor performance on texture-heavy indoor scenes (e.g., "Store"). The model's sensitivity to lighting and scale variations, coupled with its reliance on low-level patches rather than invariant descriptors, presents a clear ceiling on its potential performance despite representing a substantial improvement over global features.

## 3. Run 3: Custom Classifier

### 3.1. Task Summary
A high-performance pipeline was implemented by integrating context-aware saliency detection [1] with deep feature extraction. Saliency maps were generated and used to create enhanced images as shown in Figure 1, followed by feature extraction using a pre-trained AlexNet. Features were normalized [2], reduced via PCA (retaining 99% variance), and classified with a linear SVM. This approach, inspired by the VS-CNN architecture [3], achieved 86.87% accuracy under rigorous 5-fold stratified cross-validation, substantially outperforming a Dense SIFT + SVM baseline (69.27%).

### 3.2. Analysis
The results demonstrate the profound superiority of deep, semantically rich features over hand-crafted alternatives. The performance gap to the original VS-CNN paper [3] (97.8%) is attributed to three factors: computational constraints limiting saliency map quality [1], the use of a fixed pre-trained feature extractor without fine-tuning, and the implementation of proper cross-validation versus the paper's averaging over random splits. Nevertheless, the approach validated the core premise that guiding feature extraction with saliency provides measurable performance gains.


## References
[1] S. Goferman, L. Zelnik-Manor, and A. Tal. "Context-aware saliency detection." In _CVPR_, 2010.  
[2] D. U. Ozsahin et al. "Impact of feature scaling on machine learning models." In _AIE_, 2022.  
[3] J. Shi et al. "Scene categorization model using deep visually sensitive features." _IEEE Access_, 2019.
