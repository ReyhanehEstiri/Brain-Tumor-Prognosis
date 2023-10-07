import numpy as np
from skimage import io, color, feature

def calculate_texture_features(image_path):
    # Load the MRI image
    mri_image = io.imread(image_path, as_gray=True)

    # Binarize the image to segment the tumor
    threshold = 0.5  # Adjust this threshold based on your image characteristics
    binary_image = mri_image > threshold

    # Calculate GLCM
    glcm = feature.graycomatrix((binary_image * 255).astype('uint8'), [1], [0], 256, symmetric=True, normed=True)

    # Calculate texture features
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    energy = feature.graycoprops(glcm, 'energy')[0, 0]
    correlation = feature.graycoprops(glcm, 'correlation')[0, 0]

    return {
        "Contrast": contrast,
        "Dissimilarity": dissimilarity,
        "Homogeneity": homogeneity,
        "Energy": energy,
        "Correlation": correlation
    }

image_path = "C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y1.jpg"
texture_features = calculate_texture_features(image_path)

# Print the calculated texture features
for key, value in texture_features.items():
    print(f"{key}: {value}")
