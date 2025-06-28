import numpy as np
import cv2
import random

def create_artistic_target(grid_size, tile_size):
    height = grid_size[0] * tile_size
    width = grid_size[1] * tile_size
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            r = 100 + 155 * (y / height) * (1 - x / width)
            g = 50 + 100 * (x / width) + 50 * (y / height)
            b = 150 + 105 * (1 - y / height) * (x / width)
            img[y, x] = [r, g, b]
    
    center_x, center_y = width // 2, height // 2
    
    for angle in np.linspace(0, 2*np.pi, 100):
        for radius in range(5, min(width, height) // 4, 5):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                intensity = 1 - radius / (min(width, height) // 4)
                color = [200 * intensity, 150 * intensity, 100 * intensity]
                img[y, x] = color
    
    for i in range(20):
        fx = random.randint(tile_size, width - tile_size)
        fy = random.randint(tile_size, height - tile_size)
        size = random.randint(tile_size//2, tile_size*2)
        
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        color = [random.randint(50, 255) for _ in range(3)]
        
        if shape_type == 'circle':
            cv2.circle(img, (fx, fy), size//2, color, -1)
        elif shape_type == 'rectangle':
            cv2.rectangle(img, (fx-size//2, fy-size//2), 
                        (fx+size//2, fy+size//2), color, -1)
        elif shape_type == 'triangle':
            points = np.array([[fx, fy-size//2], 
                             [fx-size//2, fy+size//2], 
                             [fx+size//2, fy+size//2]], np.int32)
            cv2.fillPoly(img, [points], color)
    
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def extract_tiles(target_image, grid_size, tile_size):
    target_tiles = []
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_start = i * tile_size
            y_end = (i + 1) * tile_size
            x_start = j * tile_size
            x_end = (j + 1) * tile_size
            
            tile = target_image[y_start:y_end, x_start:x_end]
            target_tiles.append(tile)
    
    return target_tiles