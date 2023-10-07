import cv2

# Load the MRI image
mri_image = cv2.imread("C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y1.jpg", cv2.IMREAD_GRAYSCALE)

# Apply thresholding to segment the tumor
_, thresholded_image = cv2.threshold(mri_image, 100, 255, cv2.THRESH_BINARY)

# Invert the image to have tumor region as white
inverted_image = cv2.bitwise_not(thresholded_image)

# Specify the save path
save_path = "c:\\Users\\estir\\OneDrive\\Desktop\\کد\\Preprocess(1)\tumor_mask.jpg"  

# Save the mask
cv2.imwrite(save_path, inverted_image)