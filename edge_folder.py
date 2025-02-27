import os
from PIL import Image
import numpy as np
import sys

def save_image(image, file_path):
    # Convert numpy array to PIL Image and save it
    img = Image.fromarray(image)
    img.save(file_path)

def rgb_to_gray(image):
    # Calculate grayscale values for each pixel using average method
    gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for row in range(len(image)):
        for col in range(len(image[row])):
            gray[row][col] = np.average(image[row][col])
    return gray

def apply_gaussian_blur(image, kernel_size):
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                         np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel(kernel_size, sigma=1.0)

    rows, cols = image.shape
    k_half = kernel_size // 2
    output = np.zeros_like(image)

    for i in range(k_half, rows - k_half):
        for j in range(k_half, cols - k_half):
            output[i, j] = np.sum(image[i - k_half: i + k_half + 1, j - k_half: j + k_half + 1] * kernel)

    return output

def compute_gradient_magnitude_and_orientation(image, sobel_kernel_size):
    if sobel_kernel_size == 3:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif sobel_kernel_size == 5:
        sobel_x = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]])
        sobel_y = np.array([[-1, -2, -3, -2, -1], [-2, -3, -5, -3, -2], [0, 0, 0, 0, 0], [2, 3, 5, 3, 2], [1, 2, 3, 2, 1]])
    else:
        sys.exit("Sobel kernel size should be 3 or 5!")

    rows, cols = image.shape
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    half_size = sobel_kernel_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            gradient_x[i, j] = np.sum(window * sobel_x)
            gradient_y[i, j] = np.sum(window * sobel_y)

    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2).astype(np.uint8)
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation

def apply_non_max_suppression(magnitude, orientation):
    suppressed_magnitude = np.copy(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = orientation[i, j]
            q = [0, 0]
            if (-np.pi / 8 <= angle < np.pi / 8) or (7 * np.pi / 8 <= angle):
                q = [magnitude[i, j + 1], magnitude[i, j - 1]]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                q = [magnitude[i + 1, j + 1], magnitude[i - 1, j - 1]]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                q = [magnitude[i + 1, j], magnitude[i - 1, j]]
            else:
                q = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]

            if magnitude[i, j] < max(q):
                suppressed_magnitude[i, j] = 0

    return suppressed_magnitude

def apply_edge_tracking_by_hysteresis(magnitude, low_threshold, high_threshold):
    rows, cols = magnitude.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)

    strong_edge_i, strong_edge_j = np.where(magnitude >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))

    edge_map[strong_edge_i, strong_edge_j] = 255

    for i, j in zip(weak_edge_i, weak_edge_j):
        if (edge_map[i - 1:i + 2, j - 1:j + 2] == 255).any():
            edge_map[i, j] = 255

    return edge_map

def canny_edge_detection(image, low_threshold, high_threshold, gaussian_kernel_size=5, sobel_kernel_size=3):
    blurred_image = apply_gaussian_blur(image, gaussian_kernel_size)
    gradient_magnitude, gradient_orientation = compute_gradient_magnitude_and_orientation(blurred_image, sobel_kernel_size)
    non_max_suppressed = apply_non_max_suppression(gradient_magnitude, gradient_orientation)
    edge_map = apply_edge_tracking_by_hysteresis(non_max_suppressed, low_threshold, high_threshold)
    return edge_map

def process_images_in_folder(input_folder, output_folder, low_threshold, high_threshold, gaussian_kernel_size=5, sobel_kernel_size=3):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"Processing: {filename}")
        original_image = Image.open(input_path)
        original_image = np.array(original_image)

        if len(original_image.shape) == 3:
            original_image = rgb_to_gray(original_image)

        edge_image = canny_edge_detection(original_image, low_threshold, high_threshold, gaussian_kernel_size, sobel_kernel_size)

        output_path = os.path.join(output_folder, filename)
        save_image(edge_image, output_path)
        print(f"Saved edge-detected image: {output_path}")

# Example usage
input_folder = "Cars"
output_folder = "Edge2nd"
low_threshold = 100
high_threshold = 220

process_images_in_folder(input_folder, output_folder, low_threshold, high_threshold)
