import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the MRI image slice (replace 'image_path' with the actual path to the image)
image_path = "C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y10.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Flatten the image to a 1D array for clustering
flat_image = image.reshape((-1, 1))

# Standardize the pixel intensities
scaler = StandardScaler()
scaled_image = scaler.fit_transform(flat_image)

# Choose the number of clusters (tumor and non-tumor regions)
num_clusters = 2

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(scaled_image)

# Get cluster assignments and reshape to original image dimensions
cluster_assignments = kmeans.labels_
clustered_image = cluster_assignments.reshape(image.shape)

# Plot the original image and the clustered image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(clustered_image, cmap='viridis')
plt.title('Clustered Image')
plt.axis('off')

plt.tight_layout()
plt.show()
