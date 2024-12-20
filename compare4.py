import numpy as np
from pathlib import Path

def euclidean_distance(x, y):
    """
    Compute euclidean distance between two vectors
    """
    return np.sqrt(np.sum((x - y) ** 2))

def cdist_custom(XA, XB):
    """
    Compute distance matrix between two sets of vectors
    """
    n_a = len(XA)
    n_b = len(XB)
    distances = np.zeros((n_a, n_b))
    
    for i in range(n_a):
        for j in range(n_b):
            distances[i, j] = euclidean_distance(XA[i], XB[j])
            
    return distances

def hungarian_algorithm(cost_matrix):
    """
    Implementation of the Hungarian (Munkres) algorithm
    """
    n, m = cost_matrix.shape
    
    # Step 1: Subtract row minima
    cost_matrix = cost_matrix.copy()
    row_min = cost_matrix.min(axis=1, keepdims=True)
    cost_matrix -= row_min
    
    # Step 2: Subtract column minima
    col_min = cost_matrix.min(axis=0, keepdims=True)
    cost_matrix -= col_min
    
    # Create masks for covered rows and columns
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(m, dtype=bool)
    starred_zeros = np.zeros((n, m), dtype=bool)
    primed_zeros = np.zeros((n, m), dtype=bool)
    
    # Step 3: Initial starring of zeros
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                starred_zeros[i, j] = True
                row_covered[i] = True
                col_covered[j] = True
    
    row_covered[:] = False
    col_covered[:] = False
    
    # Cover columns with starred zeros
    for j in range(m):
        if np.any(starred_zeros[:, j]):
            col_covered[j] = True
    
    while not np.all(col_covered):
        # Find uncovered zero
        zero_found = False
        for i in range(n):
            for j in range(m):
                if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                    primed_zeros[i, j] = True
                    zero_found = True
                    break
            if zero_found:
                break
        
        if not zero_found:
            # Find smallest uncovered value
            min_val = float('inf')
            for i in range(n):
                for j in range(m):
                    if not row_covered[i] and not col_covered[j]:
                        min_val = min(min_val, cost_matrix[i, j])
            
            # Add min_val to covered rows, subtract from uncovered columns
            for i in range(n):
                for j in range(m):
                    if row_covered[i]:
                        cost_matrix[i, j] += min_val
                    if not col_covered[j]:
                        cost_matrix[i, j] -= min_val
            continue
        
        # Find starred zero in prime's row
        starred_col = -1
        for j in range(m):
            if starred_zeros[i, j]:
                starred_col = j
                break
        
        if starred_col == -1:
            # Augmenting path starting at the primed zero
            path = [(i, j)]
            while True:
                # Find starred zero in current column
                row = -1
                for k in range(n):
                    if starred_zeros[k, path[-1][1]]:
                        row = k
                        break
                if row == -1:
                    break
                path.append((row, path[-1][1]))
                
                # Find primed zero in current row
                col = -1
                for k in range(m):
                    if primed_zeros[row, k]:
                        col = k
                        break
                path.append((row, col))
            
            # Augment path
            for r, c in path:
                starred_zeros[r, c] = not starred_zeros[r, c]
            
            primed_zeros[:] = False
            row_covered[:] = False
            col_covered[:] = False
            
            # Cover columns with starred zeros
            for j in range(m):
                if np.any(starred_zeros[:, j]):
                    col_covered[j] = True
        else:
            # Cover row of primed zero and uncover column of starred zero
            row_covered[i] = True
            col_covered[starred_col] = False
    
    row_ind, col_ind = np.where(starred_zeros)
    return row_ind, col_ind

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
    
    cost_matrix = cdist_custom(sc1, sc2)
    row_ind, col_ind = hungarian_algorithm(cost_matrix)
    
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return total_cost

def is_edge_image(img):
    black_ratio = (img == 0).sum() / img.size
    return black_ratio > 0.8

def compare_with_folder(target_image_path, folder_path="EdgeDetectedFolder"):
    # Read target image as numpy array
    target_img = np.load(target_image_path)  # Assuming .npy format
    
    results = []
    folder = Path(folder_path)
    
    # Get all numpy files from the folder
    valid_extensions = ('.npy',)
    total_files = len([f for f in folder.glob('*') if f.suffix.lower() in valid_extensions])
    processed_files = 0
    
    print("\nComparing images...")
    print("-" * 60)
    print(f"{'Image Name':<40} {'Score':<15} {'Progress'}")
    print("-" * 60)
    
    for img_path in folder.glob('*'):
        if img_path.suffix.lower() in valid_extensions:
            try:
                comp_img = np.load(str(img_path))
                similarity_score = compare_shapes(target_img, comp_img)
                results.append((img_path.name, similarity_score))
                
                processed_files += 1
                progress = f"{processed_files}/{total_files}"
                print(f"{img_path.name:<40} {similarity_score:<15.2f} {progress}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
    
    results.sort(key=lambda x: x[1])
    return results

def main():
    target_image = "target.npy"  # Replace with your target numpy array file
    
    print("\nComparing images...")
    all_results = compare_with_folder(target_image)
    
    print("\n\nFinal Results (Most similar to least similar):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Image Name':<40} {'Score'}")
    print("-" * 60)
    
    for i, (filename, score) in enumerate(all_results, 1):
        print(f"{i:<6} {filename:<40} {score:.2f}")
    
    print("\nSummary:")
    print("-" * 60)
    print(f"Total images compared: {len(all_results)}")
    print(f"Most similar score: {all_results[0][1]:.2f}")
    print(f"Least similar score: {all_results[-1][1]:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()