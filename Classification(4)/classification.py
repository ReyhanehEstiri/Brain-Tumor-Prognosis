import numpy as np

def detect_tumor(image_path):
    return np.random.randint(0, 2)

def get_image_size(image_path):

    width = np.random.randint(100, 500)
    height = np.random.randint(100, 500)
    return (width, height)

def get_image_center(image_path):

    return (np.random.randint(50, 200), np.random.randint(50, 200))

def get_tumor_periphery(image_path):

    return np.random.randint(10, 50)

def classify_image(image_path):
    # Extract features
    size = get_image_size(image_path)
    center = get_image_center(image_path)
    tumor_periphery = get_tumor_periphery(image_path)
    tumor_present = detect_tumor(image_path)

    # Classify based on features
    if size[0] > 300 and tumor_periphery > 30:
        return "Malignant"
    elif size[0] < 200 and tumor_present == 0:
        return "Benign"
    elif center[0] < 100 and center[1] < 100:
        return "Benign"
    else:
        return "Uncertain"

# Example usage
image_path ="C:\\Users\\estir\\OneDrive\\Desktop\\کد\\Preprocess(1)\\NLM'.png"

classification = classify_image(image_path)
print(f"The image is classified as: {classification}")
