import cv2
import matplotlib.pyplot as plt

def non_local_means_denoising(image_path, output_path, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    noisy_image = cv2.imread(image_path)
    
    if noisy_image is None:
        print("Error: Could not load the image.")
        return
    
    gray_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h, templateWindowSize, searchWindowSize)
    
    cv2.imwrite(output_path, denoised_image)

input_image_path = "C:\\Users\\estir\\OneDrive\\Desktop\\Brain-Tumors-main\\data\\yes\\Y1.jpg"
output_image_path = 'denoised_image.jpg'

h = 10
hColor = 10
templateWindowSize = 7
searchWindowSize = 21

non_local_means_denoising(input_image_path, output_image_path, h, hColor, templateWindowSize, searchWindowSize)

# Display the noisy and denoised images
noisy_image = cv2.imread(input_image_path)
denoised_image = cv2.imread(output_image_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
plt.title('Denoised Image')
plt.axis('off')

plt.tight_layout()
plt.show()
