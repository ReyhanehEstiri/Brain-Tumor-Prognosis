import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import requests
from io import BytesIO

# Replace 'image_url' with the actual URL of the image you want to load
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTXpokLVBokg046J12GTKFCCHcTWBuA_bXYWA&usqp=CAU'

# Load the image from the URL using Pillow and requests
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Convert the Pillow image to a NumPy array
image_array = np.array(image)

# Apply a median filter with a specified neighborhood size
filtered_image = ndimage.median_filter(image_array, size=3)  # Adjust the size as needed

# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()
