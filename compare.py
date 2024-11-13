from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# Define the target size for resizing (width, height)
target_size = (400, 400)

# Define paths to the images of the 10 Ford models
model_image_paths = [
    'Cars/focus2016.jpg',  # Replace with your image paths
    'Cars/focus2017.png',
]

# Function to preprocess the image and mask the background
def preprocess_image(image_path):
    # Load, resize, and convert to grayscale
    image = Image.open(image_path).resize(target_size).convert('L')
    
    # Apply edge detection
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Apply a threshold to create a binary mask for the car
    threshold_value = 100  # Adjust this value to separate the car from the background
    mask = np.array(edges) > threshold_value  # Binary mask of the car
    
    # Apply the mask to the edge-detected image
    edges_array = np.array(edges)
    masked_edges = np.where(mask, edges_array, 0)  # Zero out background edges
    
    return masked_edges

# Store edge-detected images with background removed for each model
model_edges = [(preprocess_image(path), path) for path in model_image_paths]

# Function to find the best match with the lowest MSE
def find_best_match(test_edges, model_edges):
    best_score = float('inf')
    best_match_path = ""
    for model_edge, path in model_edges:
        mse = np.mean((test_edges - model_edge) ** 2)
        if mse < best_score:
            best_score = mse
            best_match_path = path
    return best_score, best_match_path

# Load a test image and preprocess it to remove background edges
test_image_path = 'Cars/focus2016.jpg'  # Replace with the path of your test image
test_edges = preprocess_image(test_image_path)

# Find the best match
best_score, best_match_path = find_best_match(test_edges, model_edges)

# Display the test image and the best match model
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_edges, cmap='gray')
plt.title("Test Image Edges")

plt.subplot(1, 2, 2)
best_match_image = Image.open(best_match_path).resize(target_size)
plt.imshow(best_match_image, cmap='gray')
plt.title(f"Best Match: {best_match_path} (MSE: {best_score:.2f})")

plt.show()
