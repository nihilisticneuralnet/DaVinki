import numpy as np
import cv2
import os
from config import Config
from dataset_loader import load_cifar_dataset
from image_utils import create_artistic_target, extract_tiles
from feature_extraction import compute_all_features
from spatial_aware_algo import unique_spatial_aware_assignment
from mosaic_generator import create_mosaic_from_assignment, calculate_assignment_cost
from visualization import display_results
from utils import save_results
from scipy.spatial.distance import cdist

class DaVinki:
    def __init__(self, target_image_path, tile_size=32, grid_size=(64, 64)):
        self.config = Config(target_image_path, tile_size, grid_size)
        
        self.target_image = None
        self.target_tiles = []
        self.cifar_images = []
        self.cifar_features = []
        self.target_features = []
        
    def load_cifar_dataset(self, num_samples=10000):
        self.cifar_images, self.cifar_labels = load_cifar_dataset(num_samples, self.config.grid_size)
        
    def load_target_image(self):
        print("Loading target image...")
        
        if not os.path.exists(self.config.target_image_path):
            print("Creating sample target image...")
            self.target_image = create_artistic_target(self.config.grid_size, self.config.tile_size)
        else:
            img = cv2.imread(self.config.target_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target_size = (self.config.grid_size[1] * self.config.tile_size, 
                          self.config.grid_size[0] * self.config.tile_size)
            self.target_image = cv2.resize(img, target_size)
        
        print(f"Target image shape: {self.target_image.shape}")
        
    def extract_tiles(self):
        print("Extracting tiles from target image...")
        self.target_tiles = extract_tiles(self.target_image, self.config.grid_size, self.config.tile_size)
        print(f"Extracted {len(self.target_tiles)} tiles")
        
    def compute_all_features(self):
        self.target_features, self.cifar_features = compute_all_features(
            self.target_tiles, self.cifar_images, self.config.tile_size
        )
        
    def run_unique_spatial_mosaic(self):        
        assignment = unique_spatial_aware_assignment(
            self.target_features, self.cifar_features, self.config.grid_size
        )
        mosaic = create_mosaic_from_assignment(
            assignment, self.cifar_images, self.config.grid_size, self.config.tile_size
        )
        
        distance_matrix = cdist(self.target_features, self.cifar_features, metric='euclidean')
        cost = calculate_assignment_cost(assignment, distance_matrix)
        unique_images = len(set(assignment))
        mse = np.mean((self.target_image.astype(float) - mosaic.astype(float)) ** 2)
        
        results = {
            'assignment': assignment,
            'mosaic': mosaic,
            'unique_images': unique_images,
            'total_tiles': len(assignment)
        }
        
        print(f"Results:")
        print(f"  Unique images used: {unique_images}")
        print(f"  Total tiles: {len(assignment)}")
        
        display_results(self.target_image, results)
        
        return results


def main():
    generator = DaVinki(
        target_image_path='path', # insert the image path
        tile_size=32, 
        grid_size=(64, 64)
    )
  
    print(f"Target image: {generator.config.target_image_path}")
    print(f"Tile size: {generator.config.tile_size}x{generator.config.tile_size}")
    print(f"Grid size: {generator.config.grid_size}")
    print("-" * 40)
    
    # Step 1: load cifar-10 dataset
    print("Step 1: Loading CIFAR-10 dataset...")
    generator.load_cifar_dataset(num_samples=10000)
    print(f"Loaded {len(generator.cifar_images)} CIFAR images")
    
    # Step 2: load and process target image
    print("\nStep 2: Loading target image...")
    generator.load_target_image()
    
    # Step 3: extract tiles from target image
    print("\nStep 3: Extracting tiles...")
    generator.extract_tiles()
    
    # Step 4: compute features for all images
    print("\nStep 4: Computing features...")
    generator.compute_all_features()
    print(f"Target features shape: {generator.target_features.shape}")
    print(f"CIFAR features shape: {generator.cifar_features.shape}")
    
    # Step 5: generate mosaic using spatial-aware algorithm
    print("\nStep 5: Generating mosaic...")
    results = generator.run_unique_spatial_mosaic()
    
    print("\nStep 6: Saving results...")
    save_results(results, generator.config)
    
    print(f"Final mosaic uses {results['unique_images']} unique images")
    print(f"out of {results['total_tiles']} total tiles")
    print(f"Uniqueness ratio: {results['unique_images']/results['total_tiles']:.2%}")
    
        
if __name__ == "__main__":
    main()
