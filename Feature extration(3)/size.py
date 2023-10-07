import numpy as np
from skimage import measure, io

def calculate_tumor_features(image_path):
    # Load the MRI image
    mri_image = io.imread(image_path, as_gray=True)
    
    # Binarize the image to segment the tumor
    threshold = 0.5  # Adjust this threshold based on your image characteristics
    binary_image = mri_image > threshold
    
    # Label connected components
    labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
    
    # Calculate the area of each labeled region (in pixels)
    region_areas = measure.regionprops(labeled_image)
    
    # Find the largest region (assuming it's the tumor)
    largest_region = max(region_areas, key=lambda region: region.area)
    
    # Calculate tumor size (in pixels)
    tumor_size = largest_region.area
    
    # Calculate tumor circumference (in pixels)
    tumor_perimeter = largest_region.perimeter
    
    # Calculate tumor centroid coordinates
    centroid = largest_region.centroid
    
    return tumor_size, tumor_perimeter, centroid

image_path = "C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y1.jpg"  
tumor_size, tumor_perimeter, centroid = calculate_tumor_features(image_path)

print(f"Tumor size: {tumor_size} pixels")
print(f"Tumor circumference: {tumor_perimeter} pixels")
print(f"Tumor centroid coordinates: {centroid}")
