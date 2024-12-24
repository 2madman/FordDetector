import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from canny import canny_edge_detection, rgb_to_gray, save_image
from PIL import Image

'''
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
'''


def get_points_from_edge(edge_img, n_points=100):
    """
    Enhanced edge point sampling that attempts to maintain shape structure
    by walking along the edge and sampling uniformly
    """
    y_coords, x_coords = np.where(edge_img > 0)
    if len(x_coords) == 0:
        return np.array([])
    
    # Start with the leftmost point
    start_idx = np.argmin(x_coords)
    points = [(x_coords[start_idx], y_coords[start_idx])]
    used = {(x_coords[start_idx], y_coords[start_idx])}
    
    # Find connected points using 8-connectivity
    while len(points) < min(n_points, len(x_coords)):
        last_x, last_y = points[-1]
        candidates = []
        
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                new_x, new_y = last_x + dx, last_y + dy
                if (new_x, new_y) in zip(x_coords, y_coords) and (new_x, new_y) not in used:
                    dist = np.sqrt(dx*dx + dy*dy)
                    candidates.append((dist, (new_x, new_y)))
        
        if not candidates:
            # If no connected points, find the nearest unused point
            unused_coords = [(x, y) for x, y in zip(x_coords, y_coords) if (x, y) not in used]
            if not unused_coords:
                break
                
            dists = [(np.sqrt((x-last_x)**2 + (y-last_y)**2), (x, y)) 
                     for x, y in unused_coords]
            _, (next_x, next_y) = min(dists)
        else:
            _, (next_x, next_y) = min(candidates)
            
        points.append((next_x, next_y))
        used.add((next_x, next_y))
    
    # If we need more points, interpolate
    if len(points) < n_points:
        points = np.array(points)
        # Linear interpolation between existing points
        new_points = []
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            new_points.append(p1)
            # Add interpolated points
            for t in np.linspace(0, 1, num=n_points//len(points))[1:-1]:
                new_point = p1 + t * (p2 - p1)
                new_points.append(new_point)
        new_points.append(points[-1])
        points = np.array(new_points)
    elif len(points) > n_points:
        # Uniform sampling if we have too many points
        indices = np.linspace(0, len(points)-1, n_points, dtype=int)
        points = np.array([points[i] for i in indices])
    
    return np.array(points)

def compute_shape_context(points, n_bins_r=12, n_bins_theta=24, r_inner=0.1, r_outer=3.0):
    """
    Enhanced shape context computation with improved binning and normalization
    """
    n_points = len(points)
    if n_points == 0:
        return np.array([])
    
    # Center and normalize points
    center = np.mean(points, axis=0)
    points = points - center
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    
    # Improved logarithmic radial bins
    r_bins = np.logspace(np.log10(r_inner), np.log10(r_outer), n_bins_r)
    theta_bins = np.linspace(0, 2*np.pi, n_bins_theta+1)
    
    shape_contexts = []
    for i in range(n_points):
        # Compute relative coordinates
        diff = points - points[i]
        dists = np.sqrt((diff ** 2).sum(axis=1))
        angles = np.arctan2(diff[:, 1], diff[:, 0]) % (2*np.pi)
        
        # Remove self-reference point
        mask = dists > 0
        dists = dists[mask]
        angles = angles[mask]
        
        if len(dists) > 0:
            # Local scale normalization
            mean_dist = np.mean(dists)
            dists = dists / mean_dist
            
            # Compute weighted histogram
            hist, _, _ = np.histogram2d(
                dists, angles,
                bins=[r_bins, theta_bins],
                weights=1/np.sqrt(dists)  # Give more weight to closer points
            )
            
            # Normalize histogram
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            
        else:
            hist = np.zeros((n_bins_r, n_bins_theta))
        
        shape_contexts.append(hist.flatten())
    
    return np.array(shape_contexts)

def compare_shapes(img1, img2, n_points=200, lambda_weight=0.5):
    """
    Enhanced shape comparison with weighted cost and improved matching
    """
    points1 = get_points_from_edge(img1, n_points)
    points2 = get_points_from_edge(img2, n_points)
    
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    sc1 = compute_shape_context(points1)
    sc2 = compute_shape_context(points2)
    
    # Compute chi-square distance for histograms
    def chi2_distance(h1, h2):
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        diff_sq = (h1 - h2) ** 2
        sum_arr = h1 + h2
        # Enhanced chi-square distance with stronger penalty
        base_chi2 = np.sum(diff_sq / (sum_arr + eps))
        # Add L1 norm component to increase sensitivity
        l1_diff = np.sum(np.abs(h1 - h2))
        return base_chi2 * (1 + 0.2 * l1_diff)
    
    # Compute shape context cost matrix
    cost_sc = np.zeros((len(sc1), len(sc2)))
    for i in range(len(sc1)):
        for j in range(len(sc2)):
            cost_sc[i, j] = chi2_distance(sc1[i], sc2[j])
    
    # Compute point location cost
    cost_points = cdist(points1, points2, metric='euclidean')
    
    # Combine costs with weighting
    cost_matrix = (1 - lambda_weight) * cost_sc + lambda_weight * cost_points
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Compute final weighted cost
    total_cost = cost_matrix[row_ind, col_ind].sum() / len(row_ind)
    
    return total_cost

def find_best_rotation(points1, points2, n_rotations=36):
    """
    Helper function to find best rotation alignment between shapes
    """
    best_cost = float('inf')
    best_rotation = 0
    
    for angle in np.linspace(0, 2*np.pi, n_rotations):
        # Create rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate points1
        rotated_points = np.dot(points1, R)
        
        # Compute cost
        cost = np.mean(np.min(cdist(rotated_points, points2), axis=1))
        
        if cost < best_cost:
            best_cost = cost
            best_rotation = angle
            
    return best_rotation

def compare_with_folder(target_image_path, folder_path="Edge2nd"):
    
    target_img = Image.open(target_image_path)
    target_img = np.array(target_img)

    if len(target_img.shape) == 3:
        print("Converting to grayscale...")
        target_img = rgb_to_gray(target_img)
    save_image(target_img, "Temp/1-grayscale.jpg")
    

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
                comp_img = np.array(Image.open(str(img_path)))

                # Compare images
                similarity_score = compare_shapes(target_img, comp_img)
                results.append((img_path.name, similarity_score))
                
                # Update progress
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
    
    # Sort results by similarity score (lower is more similar)
    results.sort(key=lambda x: x[1])
    for i, (filename, score) in enumerate(results, 1):
        print(f"{i:<6} {filename:<40} {score:.2f}")
    return results

def main():
    target_image = "Cars/courier.png"  # Replace with your target image path
    
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