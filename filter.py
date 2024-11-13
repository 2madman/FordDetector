import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_car_manual(image_path):
    """
    Filter potential car regions using color and edge detection
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to different color spaces for better filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create masks for common car colors
    # Gray/Silver/White cars
    lower_gray = np.array([0, 0, 100])
    upper_gray = np.array([180, 30, 255])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Black cars
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine color masks
    color_mask = cv2.bitwise_or(mask_gray, mask_black)
    
    # Edge detection
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby contours
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for large contours (potential cars)
    contour_mask = np.zeros_like(gray)
    min_contour_area = 1000  # Adjust this value based on your image size
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    
    # Combine color and contour masks
    final_mask = cv2.bitwise_and(color_mask, contour_mask)
    
    # Clean up the mask
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask)
    
    # Add white background
    background = np.ones_like(image_rgb) * 255
    background_mask = cv2.bitwise_not(final_mask)
    background = cv2.bitwise_and(background, background, mask=background_mask)
    result = cv2.add(result, background)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(color_mask, cmap='gray')
    plt.title('Color Mask')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(contour_mask, cmap='gray')
    plt.title('Contour Mask')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(final_mask, cmap='gray')
    plt.title('Final Mask')
    plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(result)
    plt.title('Filtered Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Replace with your image path
        input_image = "Cars/focus2016.jpg"
        filter_car_manual(input_image)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()