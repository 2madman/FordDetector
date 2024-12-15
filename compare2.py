import os
import math

def read_image_raw(image_path):
    """
    Read an image file as raw bytes.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bytes: Raw image bytes
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return b''

def extract_edge_features(image_bytes):
    """
    Extract structural features from edge-detected image bytes.
    
    Args:
        image_bytes (bytes): Raw image bytes
    
    Returns:
        list: Extracted structural features
    """
    # Strategy for feature extraction from edge-detected image bytes
    features = []
    
    # Sample bytes at regular intervals to capture structural information
    sample_points = 64  # Adjust based on image size and detail needed
    step = max(1, len(image_bytes) // sample_points)
    
    for i in range(0, len(image_bytes), step):
        try:
            # Look at groups of bytes to capture structural patterns
            group = image_bytes[i:i+4]
            
            # Convert bytes to an integer representation
            if len(group) == 4:
                feature = int.from_bytes(group, byteorder='big')
                features.append(feature)
            elif len(group) > 0:
                # Pad shorter groups
                feature = int.from_bytes(group + b'\x00' * (4 - len(group)), byteorder='big')
                features.append(feature)
        except Exception:
            continue
    
    return features

def calculate_structural_similarity(features1, features2):
    """
    Calculate similarity between structural features.
    
    Args:
        features1 (list): Features of first image
        features2 (list): Features of second image
    
    Returns:
        float: Similarity score (0-1, where 1 is most similar)
    """
    # Normalize feature lengths
    min_length = min(len(features1), len(features2))
    features1 = features1[:min_length]
    features2 = features2[:min_length]
    
    # Calculate cosine similarity of features
    dot_product = sum(a * b for a, b in zip(features1, features2))
    
    magnitude1 = math.sqrt(sum(a * a for a in features1))
    magnitude2 = math.sqrt(sum(b * b for b in features2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Cosine similarity with additional structural matching score
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim

def find_most_similar_edge_image(reference_image_path, image_folder):
    """
    Find the most similar edge-detected image in a folder.
    
    Args:
        reference_image_path (str): Path to the reference image
        image_folder (str): Path to the folder containing images to compare
    
    Returns:
        str: Path to the most similar image
    """
    # Extract features from reference image
    reference_image_bytes = read_image_raw(reference_image_path)
    if not reference_image_bytes:
        raise ValueError("Could not read reference image")
    
    reference_features = extract_edge_features(reference_image_bytes)
    
    # Store similarities
    similarities = {}
    
    # Iterate through images in the folder
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        
        # Skip if it's not a file or is the reference image itself
        if not os.path.isfile(image_path) or image_path == reference_image_path:
            continue
        
        # Read and extract features of current image
        current_image_bytes = read_image_raw(image_path)
        if not current_image_bytes:
            continue
        
        current_features = extract_edge_features(current_image_bytes)
        
        # Calculate structural similarity
        similarity = calculate_structural_similarity(reference_features, current_features)
        similarities[image_path] = similarity
    
    # Find the most similar image
    if not similarities:
        return None
    
    most_similar_image = max(similarities, key=similarities.get)
    print("Similarity scores:", {os.path.basename(k): round(v, 4) for k, v in similarities.items()})
    
    return most_similar_image

# Example usage
if __name__ == "__main__":
    try:
        reference_image = "CompareCars/ranger1.jpg"
        image_folder = "EdgeDetectedFolder"
        
        most_similar = find_most_similar_edge_image(reference_image, image_folder)
        
        if most_similar:
            print(f"Most similar image: {most_similar}")
        else:
            print("No similar images found")
    
    except Exception as e:
        print(f"An error occurred: {e}")