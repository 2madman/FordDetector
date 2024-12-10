import os
import numpy as np
from PIL import Image

def resize(img, size):

    return [[[
        img[int(len(img) * i / size[0])][int(len(img[0]) * j / size[1])][k]
        for k in range(3)
    ] for j in range(size[1])] for i in range(size[0])]



def batch_resize_images(input_folder, output_folder, target_size=(224, 224)):
    """
    Resize all images in an input folder and save to an output folder.
    
    Args:
    input_folder (str): Path to the folder containing input images
    output_folder (str): Path to the folder where resized images will be saved
    target_size (tuple): Desired output size for images (width, height)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Full path to the input image
            input_path = os.path.join(input_folder, filename)
            
            try:
                # Open the image using PIL
                img = Image.open(input_path)
                
                # Convert image to numpy array
                img_array = np.array(img)
                
                # Resize the image
                resized_img_array = resize(img_array, target_size)
                
                # Convert back to PIL Image
                resized_img = Image.fromarray(resized_img_array.astype('uint8'))
                
                # Prepare output path
                output_path = os.path.join(output_folder, filename)
                
                # Save the resized image
                resized_img.save(output_path)
                
                print(f"Resized {filename} to {target_size}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
if __name__ == "__main__":
    input_folder = "Cars/"
    output_folder = "Resized/"
    
    # Optional: specify custom size
    batch_resize_images(input_folder, output_folder, target_size=(224, 224))