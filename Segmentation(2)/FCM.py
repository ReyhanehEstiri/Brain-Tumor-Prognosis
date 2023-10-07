import numpy as np
import skfuzzy as fuzz
from PIL import Image
import matplotlib.pyplot as plt

def fuzzy_cmeans_segmentation(image, num_clusters, fuzziness):
    img_gray = image.convert("L")  # Convert to grayscale
    img_array = np.array(img_gray)
    img_flatten = img_array.flatten()

    # Apply Fuzzy C-Means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        img_flatten[np.newaxis, :], num_clusters, fuzziness, error=0.005, maxiter=1000
    )

    # Get cluster centers and memberships
    img_segmented = np.argmax(u, axis=0).reshape(img_array.shape)

    return img_segmented, cntr

# Load the image
image_path = "C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y10.jpg" 
image = Image.open(image_path)

# Parameters
num_clusters = 2  # Number of clusters (regions)
fuzziness = 2.0  # Fuzziness parameter (higher values make memberships more fuzzy)

# Perform Fuzzy C-Means segmentation
segmented_image, cluster_centers = fuzzy_cmeans_segmentation(image, num_clusters, fuzziness)

# Display the original and segmented images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()
