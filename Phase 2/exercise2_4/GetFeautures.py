import cv2
import numpy as np
from skimage import io, feature, color
import matplotlib.pyplot as plt

# Load the image
image_path = 'Fruit.png'
image = io.imread(image_path)

# Check if the image has an alpha channel and remove it if necessary
if image.shape[2] == 4:
    image = image[:, :, :3]  # Discard the alpha channel

# Convert the image to grayscale
gray_image = color.rgb2gray(image)
gray_image = (gray_image * 255).astype(np.uint8)  # Convert to integer type

# 1. Color Histogram
def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extracts a color histogram from the image.
    """
    # Compute a 3D color histogram in the RGB color space
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

color_hist = extract_color_histogram(image)

# 2. Edge Detection using Canny
def extract_edges(image, sigma=1.0):
    """
    Extracts edges from the image using the Canny edge detector.
    """
    edges = feature.canny(image, sigma=sigma)
    return edges

edges = extract_edges(gray_image)

# 3. Texture using Local Binary Patterns (LBP)
def extract_lbp(image, P=8, R=1.0):
    """
    Extracts Local Binary Patterns (LBP) from the image.
    """
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist, lbp

lbp_hist, lbp_image = extract_lbp(gray_image)

# 4. Key Points and Descriptors using ORB
def extract_keypoints_and_descriptors(image):
    """
    Extracts keypoints and descriptors using ORB.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

keypoints, descriptors = extract_keypoints_and_descriptors(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

# Display results
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')

# Edges
plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

# LBP Image
plt.subplot(2, 2, 3)
plt.imshow(lbp_image, cmap='gray')
plt.title('Local Binary Patterns (LBP)')

# Key Points
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
plt.subplot(2, 2, 4)
plt.imshow(output_image)
plt.title('Key Points using ORB')

plt.show()

# Display LBP histogram
plt.figure()
plt.bar(np.arange(len(lbp_hist)), lbp_hist, width=0.5, color='gray')
plt.title('LBP Histogram')
plt.show()

# Print extracted features
print("Color Histogram:", color_hist)
print("LBP Histogram:", lbp_hist)
print("Number of Keypoints:", len(keypoints))
print("Descriptors Shape:", descriptors.shape)
