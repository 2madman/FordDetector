import os
from PIL import Image
import numpy as np
import canny
import re

def extract_until_number(input_string):
    # Get the portion after the last '/'
    last_section = input_string.split("/")[-1]
    # Extract characters until the first number
    match = re.match(r"^[^\d]*", last_section)
    return match.group(0) if match else ""

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


def extract_until_number(input_string):
    result = ""
    for char in input_string:
        if char.isdigit():  # Check if the character is a digit
            break          # Stop if a digit is found
        result += char     # Append non-digit characters to the result
    return result

def resize_image(image_array, size):
    """
    Resizes a single image represented as a numpy array to the given size.

    Parameters:
        image_array (numpy.ndarray): The input image as a numpy array.
        size (tuple): The desired output size (width, height).
    
    Returns:
        numpy.ndarray: The resized image as a numpy array.
    """
    height, width, _ = image_array.shape
    new_height, new_width = size
    
    # Create a new array for the resized image
    resized_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Resize logic: mapping original image pixels to the new size
    for i in range(new_height):
        for j in range(new_width):
            # Mapping the coordinates
            orig_i = int(i * height / new_height)
            orig_j = int(j * width / new_width)
            resized_array[i, j] = image_array[orig_i, orig_j]
    
    return resized_array




if __name__ == "__main__":
    input_image_path = "Cars/kuga6.png"  # Replace with your image path
    edge_detected_folder = "EdgeDetectedFolder"

    img = Image.open(input_image_path)
    img_array = np.array(img)
    resized_image_array = resize_image(img_array, [376,668])
    resized_image = Image.fromarray(resized_image_array)
    input_path = 'Temp/resized.png'
    resized_image.save(input_path)


    edge_detected_results = compare_with_edge_detected_folder(input_path, edge_detected_folder)
    
    lowestMse = 100000
    car = ""
    for image_path, mse in edge_detected_results.items():
        if mse< lowestMse:
            lowestMse = mse
            car = image_path
        #print(f"{image_path}: MSE={mse}")

    car = extract_until_number(car)
    print(car)