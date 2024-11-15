from PIL import Image, ImageFilter
import numpy as np

def isolate_car(image_path, threshold=120):
    # Load the image and convert it to grayscale for simplicity
    img = Image.open(image_path)
    grayscale_img = img.convert("L")  # Convert to grayscale

    # Apply edge detection to find boundaries
    edges = grayscale_img.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges)

    # Threshold the edges to make them binary (black and white)
    binary_edges = (edges_array > threshold).astype(np.uint8) * 255

    # Create an empty mask
    mask = Image.fromarray(binary_edges).convert("1")  # Convert to binary mask

    # Apply the mask to the original image
    car_only = Image.composite(img, Image.new("RGB", img.size), mask)

    return car_only

# Example usage:
image_path = 'Cars/focus.png'
car_only_image = isolate_car(image_path)
car_only_image.show()  # Show the isolated car
car_only_image.save("Cars/focusnew.png")  # Save the result if desired
