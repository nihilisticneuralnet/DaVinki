import numpy as np
import cv2
from scipy.spatial.distance import cdist

def create_mosaic_from_assignment(assignment, cifar_images, grid_size, tile_size):
    mosaic_height = grid_size[0] * tile_size
    mosaic_width = grid_size[1] * tile_size
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    
    for i, cifar_idx in enumerate(assignment):
        row = i // grid_size[1]
        col = i % grid_size[1]
        
        y_start = row * tile_size
        y_end = (row + 1) * tile_size
        x_start = col * tile_size
        x_end = (col + 1) * tile_size
        
        cifar_img = cifar_images[cifar_idx].copy()
        if cifar_img.shape[:2] != (tile_size, tile_size):
            cifar_img = cv2.resize(cifar_img, (tile_size, tile_size))
        
        mosaic[y_start:y_end, x_start:x_end] = cifar_img
    
    return mosaic

def calculate_assignment_cost(assignment, distance_matrix):
    total_cost = 0
    for tile_idx, cifar_idx in enumerate(assignment):
        total_cost += distance_matrix[tile_idx, cifar_idx]
    return total_cost