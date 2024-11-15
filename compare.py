import os
from PIL import Image
import numpy as np
import canny

def calculate_mse(image1, image2):
    # Ensure both images are numpy arrays
    img1_array = np.asarray(image1, dtype=np.float64)
    img2_array = np.asarray(image2, dtype=np.float64)
    
    # Ensure the images have the same shape
    if img1_array.shape != img2_array.shape:
        raise ValueError("Images must have the same dimensions for comparison.")
    
    # Calculate MSE
    mse = np.mean((img1_array - img2_array) ** 2)
    return mse


def compare_with_edge_detected_folder(image_path, edge_detected_folder="EdgeDetected"):
    # Open the image to compare
    with Image.open(image_path) as img:
        base_image = img.convert("L")  # Convert to grayscale for comparison

    results = {}
    
    base_image = Image.open(image_path)
    base_image = np.array(base_image)

    if len(base_image.shape) == 3:
        print("Converting to grayscale...")
        base_image = canny.rgb_to_gray(base_image)

    canny.save_image(base_image, "Temp/1-grayscale.jpg")

    low_threshold = 100
    high_threshold = 220
    gaussian_kernel_size = 5
    sobel_kernel_size = 3
    base_image = canny.canny_edge_detection(base_image, low_threshold, high_threshold, gaussian_kernel_size, sobel_kernel_size)

    # Ensure the 'EdgeDetected' folder exists
    if not os.path.isdir(edge_detected_folder):
        raise FileNotFoundError(f"Folder '{edge_detected_folder}' not found.")
    
    # Iterate over all files in the 'EdgeDetected' folder
    for file_name in os.listdir(edge_detected_folder):
        file_path = os.path.join(edge_detected_folder, file_name)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as edge_img:
                    edge_image = edge_img.convert("L")  # Convert to grayscale
                    mse = calculate_mse(base_image, edge_image)
                    results[file_path] = mse
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
    
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    return sorted_results


# Example Usage
if __name__ == "__main__":
    input_image_path = "Cars/test.png"  # Replace with your image path
    edge_detected_folder = "EdgeDetected"
    
    edge_detected_results = compare_with_edge_detected_folder(input_image_path, edge_detected_folder)
    
    for image_path, mse in edge_detected_results.items():
        print(f"{image_path}: MSE={mse}")
