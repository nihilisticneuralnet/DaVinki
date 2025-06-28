import numpy as np
from sklearn.datasets import fetch_openml

def load_cifar_dataset(num_samples, grid_size):
    print("Loading CIFAR-10 dataset...")
    
    total_tiles = grid_size[0] * grid_size[1]
    required_samples = max(num_samples, total_tiles * 2)
    
    try:
        cifar = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
        X, y = cifar
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        unique_classes = np.unique(y)
        samples_per_class = required_samples // len(unique_classes)
        
        selected_indices = []
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            selected = np.random.choice(class_indices, 
                                      min(samples_per_class, len(class_indices)), 
                                      replace=False)
            selected_indices.extend(selected)
        
        if len(selected_indices) < required_samples:
            remaining_indices = list(set(range(len(X))) - set(selected_indices))
            additional_needed = required_samples - len(selected_indices)
            additional_selected = np.random.choice(remaining_indices, 
                                                 min(additional_needed, len(remaining_indices)), 
                                                 replace=False)
            selected_indices.extend(additional_selected)
        
        cifar_images = X[selected_indices[:required_samples]]
        cifar_labels = y[selected_indices[:required_samples]]
        
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        cifar_images = generate_synthetic_images(required_samples)
        cifar_labels = np.random.randint(0, 10, required_samples)
    
    print(f"Loaded {len(cifar_images)} CIFAR images (Required: {total_tiles} unique tiles)")
    return cifar_images, cifar_labels

def generate_synthetic_images(num_samples):
    images = []
    patterns = ['gradient', 'stripes', 'circles', 'checker', 'noise', 'blend', 'waves', 'spiral']
    
    for i in range(num_samples):
        pattern = patterns[i % len(patterns)]
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        
        color1 = np.random.randint(0, 256, 3)
        color2 = np.random.randint(0, 256, 3)
        color3 = np.random.randint(0, 256, 3)
        
        variation = i // len(patterns)
        
        if pattern == 'gradient':
            direction = variation % 4 
            for y in range(32):
                for x in range(32):
                    if direction == 0:  
                        ratio = x / 32
                    elif direction == 1: 
                        ratio = y / 32
                    elif direction == 2:  
                        ratio = (x + y) / 64
                    else: 
                        ratio = np.sqrt((x-16)**2 + (y-16)**2) / 22
                    ratio = min(ratio, 1.0)
                    img[y, x] = color1 * (1 - ratio) + color2 * ratio
                    
        elif pattern == 'stripes':
            stripe_width = 2 + (variation % 6) 
            direction = (variation // 6) % 2 
            for pos in range(0, 32, stripe_width * 2):
                if direction == 0:  
                    img[pos:pos+stripe_width, :] = color1
                    img[pos+stripe_width:pos+stripe_width*2, :] = color2
                else:  
                    img[:, pos:pos+stripe_width] = color1
                    img[:, pos+stripe_width:pos+stripe_width*2] = color2
                    
        elif pattern == 'circles':
            num_circles = 1 + (variation % 3)  
            for circle_idx in range(num_circles):
                center_x = 8 + (circle_idx * 8) % 16
                center_y = 8 + (circle_idx * 8) % 16
                radius = 4 + (variation % 8)
                for x in range(32):
                    for y in range(32):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            img[x, y] = [color1, color2, color3][circle_idx % 3]
                            
        elif pattern == 'checker':
            check_size = 2 + (variation % 6)
            for x in range(0, 32, check_size):
                for y in range(0, 32, check_size):
                    color = color1 if ((x//check_size) + (y//check_size)) % 2 == 0 else color2
                    img[x:x+check_size, y:y+check_size] = color
                    
        elif pattern == 'noise':
            base_color = color1
            noise_level = 20 + (variation % 50)
            noise = np.random.normal(0, noise_level, (32, 32, 3))
            img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            
        elif pattern == 'blend':
            x_grad = np.linspace(0, 1, 32)
            y_grad = np.linspace(0, 1, 32)
            X, Y = np.meshgrid(x_grad, y_grad)
            
            blend_type = variation % 3
            if blend_type == 0:
                img[:, :, 0] = color1[0] * (1 - X) + color2[0] * X
                img[:, :, 1] = color1[1] * (1 - Y) + color2[1] * Y
                img[:, :, 2] = color1[2] * (X * Y) + color3[2] * (1 - X * Y)
            elif blend_type == 1:
                img[:, :, 0] = color1[0] * np.sin(X * np.pi) + color2[0] * np.cos(Y * np.pi)
                img[:, :, 1] = color1[1] * X + color2[1] * Y
                img[:, :, 2] = color3[2] * (X + Y) / 2
            else:
                img[:, :, 0] = color1[0] * (X * Y) + color2[0] * (1 - X * Y)
                img[:, :, 1] = color1[1] * np.abs(X - Y) + color2[1] * (1 - np.abs(X - Y))
                img[:, :, 2] = color3[2]
                
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        elif pattern == 'waves':
            frequency = 1 + (variation % 4)
            for x in range(32):
                for y in range(32):
                    wave_val = np.sin(x * frequency * np.pi / 16) * np.cos(y * frequency * np.pi / 16)
                    intensity = (wave_val + 1) / 2
                    img[y, x] = color1 * intensity + color2 * (1 - intensity)
                    
        elif pattern == 'spiral':
            center = (16, 16)
            for x in range(32):
                for y in range(32):
                    dx, dy = x - center[0], y - center[1]
                    angle = np.arctan2(dy, dx)
                    radius = np.sqrt(dx**2 + dy**2)
                    spiral_val = (angle + radius * 0.2 + variation) % (2 * np.pi)
                    intensity = spiral_val / (2 * np.pi)
                    img[y, x] = color1 * intensity + color2 * (1 - intensity)
        
        images.append(img)
    
    return np.array(images)