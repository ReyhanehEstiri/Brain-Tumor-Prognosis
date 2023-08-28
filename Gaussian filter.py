import cv2
import matplotlib.pyplot as plt

def apply_gaussian_filter(image_path, kernel_size=5, sigma=1.0):
    # Load the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred_image

# Input image path
input_image_path = 'C:\\Users\\estir\\Downloads\\Example-.png'

# Apply Gaussian filter
blurred_image = apply_gaussian_filter(input_image_path, kernel_size=5, sigma=1.5)

# Display original and blurred images using matplotlib
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Blurred image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')
plt.axis('off')

plt.tight_layout()
plt.show()
