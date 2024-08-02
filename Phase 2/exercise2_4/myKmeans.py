import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def initialize_centroids(X, K):
    """
    Randomly select initial centroids from the dataset.
    """
    indices = np.random.choice(X.shape[0], K, replace=False) #replace=False ensures no duplicates
    centroids = X[indices]
    return centroids

def find_closest_centroids(X, centroids):
    """
    Finds the closest centroid for each sample.
    """
    # Calculate the Euclidean distance between each point and the centroids
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    idx = np.argmin(distances, axis=1)
    return idx

def compute_centroids(X, idx, K):
    """
    Compute the mean of samples assigned to each centroid.
    """
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        points = X[idx == k]
        if points.shape[0] > 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

def run_kmeans(X, initial_centroids, max_iters):
    """
    K-means algorithm for a specified number of iterations.
    """
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def kmeans_init_centroids(X, K):
    """
    Initialize centroids randomly.
    """
    initial_centroids = initialize_centroids(X, K)
    return initial_centroids

# Load the image
image_path = 'Fruit.png'
image = io.imread(image_path)

# Size of the image
img_size = image.shape

# Normalize image values in the range 0 - 1
image = image / 255.0

# Check the number of channels in the image
if image.ndim == 2:
    image = np.stack((image,)*3, axis=-1)
elif image.shape[2] == 4:
    image = image[:, :, :3]

# Reshape the image to be a Nx3 matrix (N = num of pixels)
X = image.reshape(img_size[0] * img_size[1], 3)

# List of K values for which we want to generate compressed images
K_values = [2, 4, 8, 16, 100, 200, 500]
max_iters = 10

# Iterate over different values of K
for K in K_values:
    # Initialize the centroids randomly
    initial_centroids = kmeans_init_centroids(X, K)
    
    # Run K-Means
    centroids, idx = run_kmeans(X, initial_centroids, max_iters)
    
    # Find closest cluster members
    idx = find_closest_centroids(X, centroids)
    
    # Recover the image from the indices
    X_recovered = centroids[idx]
    
    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)
    
    # Display the compressed image in a new figure
    plt.figure()
    plt.imshow(X_recovered)
    plt.title(f'Compressed Image with K = {K}')
    plt.show()

# Display the original image in a new figure
plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.show()
