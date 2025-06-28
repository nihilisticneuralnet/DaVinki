import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def unique_spatial_aware_assignment(target_features, cifar_features, grid_size):
    print("Running Unique Spatial-aware approach ensuring each CIFAR image is used only once...")
    
    total_tiles = len(target_features)
    total_cifar = len(cifar_features)
    
    if total_cifar < total_tiles:
        raise ValueError(f"Not enough CIFAR images ({total_cifar}) for unique assignment ({total_tiles} tiles needed)")
    
    distance_matrix = cdist(target_features, cifar_features, metric='euclidean')
    
    assignment = [-1] * total_tiles 
    used_images = set()
    
    center_row, center_col = grid_size[0] // 2, grid_size[1] // 2
    spiral_order = generate_spiral_order(center_row, center_col, grid_size)
    
    print(f"Processing {len(spiral_order)} tiles in spiral order...")
    
    for tile_idx in tqdm(spiral_order, desc="Assigning tiles"):
        row = tile_idx // grid_size[1]
        col = tile_idx % grid_size[1]
        
        neighbors = get_tile_neighbors(row, col, grid_size)
        neighbor_assignments = [assignment[n] for n in neighbors if assignment[n] != -1]
        
        available_images = [i for i in range(total_cifar) if i not in used_images]
        
        if not available_images:
            raise ValueError("Ran out of unique images during assignment")
        
        modified_distances = {}
        
        for cifar_idx in available_images:
            base_distance = distance_matrix[tile_idx, cifar_idx]
            
            spatial_score = 0
            if neighbor_assignments:
                for neighbor_cifar in neighbor_assignments:
                    neighbor_similarity = np.linalg.norm(
                        cifar_features[cifar_idx] - cifar_features[neighbor_cifar]
                    )
                    optimal_similarity = 0.5 
                    similarity_penalty = abs(neighbor_similarity - optimal_similarity)
                    spatial_score += similarity_penalty
                
                spatial_score /= len(neighbor_assignments)
                
                spatial_weight = 0.2 
                modified_distance = base_distance + spatial_weight * spatial_score
            else:
                modified_distance = base_distance
                
            modified_distances[cifar_idx] = modified_distance
        
        best_match = min(modified_distances.keys(), key=lambda x: modified_distances[x])
        
        assignment[tile_idx] = best_match
        used_images.add(best_match)
    
    print(f"Assignment complete! Used {len(used_images)} unique images out of {total_cifar} available.")
    return assignment

def generate_spiral_order(start_row, start_col, grid_size):
    visited = set()
    order = []
    
    queue = [(start_row, start_col, 0)] 
    
    while queue:
        queue.sort(key=lambda x: x[2])  
        row, col, dist = queue.pop(0)
        
        if (row, col) in visited:
            continue
            
        if 0 <= row < grid_size[0] and 0 <= col < grid_size[1]:
            visited.add((row, col))
            tile_idx = row * grid_size[1] + col
            order.append(tile_idx)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) not in visited:
                    new_dist = abs(new_row - start_row) + abs(new_col - start_col)
                    queue.append((new_row, new_col, new_dist))
    
    return order

def get_tile_neighbors(row, col, grid_size):
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < grid_size[0] and 0 <= new_col < grid_size[1]:
            neighbor_idx = new_row * grid_size[1] + new_col
            neighbors.append(neighbor_idx)
    return neighbors