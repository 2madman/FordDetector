import numpy as np
from PIL import Image

def apply_sobel(image_path):
    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Get image dimensions
    height, width = img_array.shape

    # Initialize an array to hold the gradient magnitude
    gradient_magnitude = np.zeros((height, width))

    # Convolve the image with the Sobel kernels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 region
            region = img_array[i-1:i+2, j-1:j+2]

            # Apply the Sobel kernels
            gx = np.sum(sobel_x * region)
            gy = np.sum(sobel_y * region)

            # Calculate the gradient magnitude
            gradient_magnitude[i, j] = min(255, np.sqrt(gx**2 + gy**2))

    # Convert the result to an image
    edge_image = Image.fromarray(gradient_magnitude).convert("L")
    return edge_image