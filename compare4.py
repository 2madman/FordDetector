import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2
import os
from pathlib import Path
from canny import canny_edge_detection


def get_points_from_edge(edge_img, n_points=100):
    y_coords, x_coords = np.where(edge_img > 0)
    
    if len(x_coords) > n_points:
        idx = np.random.choice(len(x_coords), n_points, replace=False)
        points = np.column_stack((x_coords[idx], y_coords[idx]))
    else:
        points = np.column_stack((x_coords, y_coords))
    
    return points

def compute_shape_context(points, n_bins_r=5, n_bins_theta=12):
    n_points = len(points)
    shape_contexts = []
    
    r_inner = 0.1250
    r_outer = 2.0
    r_bins = np.logspace(np.log10(r_inner), np.log10(r_outer), n_bins_r)
    theta_bins = np.linspace(0, 2*np.pi, n_bins_theta+1)
    
    for i in range(n_points):
        diff = points - points[i]
        dists = np.sqrt((diff ** 2).sum(axis=1))
        angles = np.arctan2(diff[:, 1], diff[:, 0]) % (2*np.pi)
        
        mask = dists > 0
        dists = dists[mask]
        angles = angles[mask]
        
        if len(dists) > 0:
            mean_dist = np.mean(dists)
            dists = dists / mean_dist
        
        hist, _, _ = np.histogram2d(
            dists, angles, 
            bins=[r_bins, theta_bins]
        )
        shape_contexts.append(hist.flatten())
    
    return np.array(shape_contexts)

def compare_shapes(img1, img2, n_points=100):
    points1 = get_points_from_edge(img1, n_points)
    points2 = get_points_from_edge(img2, n_points)
    
    sc1 = compute_shape_context(points1)
    sc2 = compute_shape_context(points2)
    
    cost_matrix = cdist(sc1, sc2, metric='euclidean')
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    return total_cost

def is_edge_image(img):
    black_ratio = (img == 0).sum() / img.size
    return black_ratio > 0.8

def compare_with_folder(target_image_path, folder_path="Edge2nd"):
    # Read target image
    target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    target_img = canny_edge_detection(target_img,100,220,5,3)
    
    results = []
    
    # Get all image files from the folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    folder = Path(folder_path)
    
    # Count total files for progress tracking
    total_files = len([f for f in folder.glob('*') if f.suffix.lower() in valid_extensions])
    processed_files = 0
    
    print("\nComparing images...")
    print("-" * 60)
    print(f"{'Image Name':<40} {'Score':<15} {'Progress'}")
    print("-" * 60)
    
    # Process each image in the folder
    for img_path in folder.glob('*'):
        if img_path.suffix.lower() in valid_extensions:
            try:
                # Read and process comparison image
                comp_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                # Compare images
                similarity_score = compare_shapes(target_img, comp_img)
                results.append((img_path.name, similarity_score))
                
                # Update progress
                processed_files += 1
                progress = f"{processed_files}/{total_files}"
                
                # Print result for each image
                #print(f"{img_path.name:<40} {similarity_score:<15.2f} {progress}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
    
    # Sort results by similarity score (lower is more similar)
    results.sort(key=lambda x: x[1])
    for i, (filename, score) in enumerate(results, 1):
        print(f"{i:<6} {filename:<40} {score:.2f}")
    return results

def main():
    target_image = "Cars/puma6.png"  # Replace with your target image path
    
    print("\nComparing images...")
    all_results = compare_with_folder(target_image)
    
    print("\n\nFinal Results (Most similar to least similar):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Image Name':<40} {'Score'}")
    print("-" * 60)
    
    # Print all results sorted by similarity
    for i, (filename, score) in enumerate(all_results, 1):
        print(f"{i:<6} {filename:<40} {score:.2f}")
    

if __name__ == "__main__":
    main()