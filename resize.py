import os
from PIL import Image
import numpy as np

def resize_images_in_folder(input_folder, size):
    # Define the fixed output folder
    output_folder = os.path.abspath("Resized")
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Skip directories and non-image files
        if not os.path.isfile(input_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if not already in RGB mode
            img = img.convert("RGB")
            
            # Convert the image to a numpy array for resizing
            img_array = np.array(img)
            
            # Resize the image using the given logic
            resized_array = [[[ 
                img_array[int(len(img_array) * i / size[0])][int(len(img_array[0]) * j / size[1])][k]
                for k in range(3)
            ] for j in range(size[1])] for i in range(size[0])]
            
            # Convert the resized array back to an image
            resized_image = Image.fromarray(np.array(resized_array, dtype=np.uint8))
            
            # Save the resized image
            output_path = os.path.join(output_folder, filename)
            resized_image.save(output_path)
            print(f"Resized and saved: {output_path}")
    
    print(f"All images resized and saved in folder: {output_folder}")


resize_images_in_folder("Cars", [376, 668])
