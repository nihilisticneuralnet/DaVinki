import numpy as np
import cv2
from skimage import feature
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def extract_advanced_features(image, tile_size):
    if image.shape[:2] != (tile_size, tile_size):
        image = cv2.resize(image, (tile_size, tile_size))
    
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    features = []
    
    for channel in range(3):
        channel_data = image[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.median(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75)
        ])
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for channel in range(3):
        channel_data = hsv[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data)
        ])
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        np.mean(grad_mag),
        np.std(grad_mag),
        np.mean(np.abs(grad_x)),
        np.mean(np.abs(grad_y))
    ])
    
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    low_freq = magnitude_spectrum[center_h-h//4:center_h+h//4, 
                                center_w-w//4:center_w+w//4]
    features.append(np.mean(low_freq))
    
    high_freq = magnitude_spectrum.copy()
    high_freq[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
    features.append(np.mean(high_freq))
    
    edges = cv2.Canny(gray, 50, 150)
    features.extend([
        np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),
        len(cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    ])
    
    try:
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
        features.extend(lbp_hist)
    except:
        features.extend([0] * 10)
    
    return np.array(features)

def compute_all_features(target_tiles, cifar_images, tile_size):
    print("Computing advanced features...")
    
    target_features = []
    for tile in tqdm(target_tiles, desc="Target tiles"):
        features = extract_advanced_features(tile, tile_size)
        target_features.append(features)
    
    cifar_features = []
    for img in tqdm(cifar_images, desc="CIFAR images"):
        features = extract_advanced_features(img, tile_size)
        cifar_features.append(features)
    
    target_features = np.array(target_features)
    cifar_features = np.array(cifar_features)
    
    scaler = StandardScaler()
    target_features = scaler.fit_transform(target_features)
    cifar_features = scaler.transform(cifar_features)
    
    return target_features, cifar_features