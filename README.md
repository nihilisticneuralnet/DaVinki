# DaVinki: Photomosaic Generator using CIFAR-10 Dataset

A photomosaic generator that transforms target images into artistic mosaics using images from the **CIFAR-10 dataset**. The project uses feature extraction techniques, **spatial-aware** algorithm to create mosaics while ensuring each source image is used only once.

## Example Outputs

![Mona Lisa](/img/mona_output.png) 

![DaVini Twins Input](/img/davinki_output.png)

## Architecture

https://github.com/user-attachments/assets/2aad45f7-0669-408a-a856-cf6ab8ce478c


https://github.com/user-attachments/assets/44412361-e036-4ff5-a7f4-58ea7c84c4c3


## Key Features

### Feature Extraction
- **Multi-channel Color Statistics**: RGB and HSV color space analysis with percentile-based distributions
- **Texture Analysis**: Gradient-based texture features using Sobel operators
- **Frequency Domain Features**: FFT-based low and high frequency component analysis
- **Edge Detection**: Canny edge detection for shape and contour analysis
- **Local Binary Patterns**: Texture characterization using rotation-invariant LBP
- **Feature Normalization**: StandardScaler preprocessing for optimal matching

### Spatial-Aware Assignment Algorithm
- **Unique Image Constraint**: Each CIFAR-10 image used exactly once across the entire mosaic
- **Spiral Processing**: Center-out tile processing for optimal spatial coherence
- **Neighbor-Aware Matching**: Considers similarity relationships with adjacent tiles
- **Dynamic Distance Modification**: Balances feature similarity with spatial constraints

### Dataset Handling
- **Automatic CIFAR-10 Loading**: Fetches dataset with balanced class (stratified) sampling


## Technical Specifications

### Architecture
- **Grid-based Tiling**: Configurable grid sizes (default: 64x64)
- **Tile Resolution**: Adjustable tile size (default: 32x32 pixels)
- **Feature Vector**: 45-dimensional feature space per image
- **Distance Metric**: Euclidean distance in normalized feature space

### Performance Metrics
- **Assignment Cost**: Cumulative feature distance across all tile assignments
- **Uniqueness Ratio**: Percentage of unique images used vs total tiles
- **Mean Squared Error**: Pixel-level difference between target and mosaic
- **Processing Time**: Per-tile assignment timing for performance analysis

## Installation

### Prerequisites
- Python 3.7+
- OpenCV 4.0+
- NumPy, SciPy
- scikit-learn
- scikit-image
- matplotlib
- tqdm

### Setup
```bash
# Clone the repository
git clone https://github.com/nihilisticneuralnet/DaVinki.git
cd DaVinki

# Install dependencies
pip install -r requirements.txt

python main.py
```

## Usage

### Basic Usage
```python
from main import DaVinki

generator = DaVinki(
    target_image_path='path/to/your/image.jpg',
    tile_size=32,
    grid_size=(64, 64)
)

generator.load_cifar_dataset(num_samples=10000)
generator.load_target_image()
generator.extract_tiles()
generator.compute_all_features()
results = generator.run_unique_spatial_mosaic()
```

### Configuration Options
- **tile_size**: Size of individual mosaic tiles (8-64 pixels recommended)
- **grid_size**: Mosaic dimensions in tiles (e.g., (32, 32) for 32x32 grid)
- **num_samples**: Number of CIFAR-10 images to load (minimum: grid_size[0] × grid_size[1])


## Algorithm Details

### Feature Extraction Pipeline
1. **Color Space Transformation**: RGB → HSV conversion for perceptual color analysis
2. **Statistical Moments**: Mean, standard deviation, median, and quartiles per channel
3. **Gradient Analysis**: Sobel operator application for texture characterization
4. **Frequency Analysis**: 2D FFT for spatial frequency decomposition
5. **Edge Density**: Canny edge detection with contour counting
6. **Local Patterns**: Uniform Local Binary Pattern computation

### Spatial-Aware Assignment Process
1. **Spiral Ordering**: Generate center-out processing sequence
2. **Neighbor Context**: Identify adjacent tiles and their assignments
3. **Distance Modification**: Apply spatial coherence penalties
4. **Greedy Selection**: Choose optimal available image per tile
5. **Uniqueness Enforcement**: Track used images to prevent repetition

### Quality Metrics
- **Visual Coherence**: Measured through neighbor similarity analysis
- **Color Fidelity**: MSE between original and mosaic RGB values
- **Feature Preservation**: Distance minimization in high-dimensional space
- **Coverage Efficiency**: Ratio of unique images utilized

## File Structure

```
DaVinki/
├── src/main.py                 # Main execution script
├── src/config.py               # Configuration management
├── src/dataset_loader.py       # CIFAR-10 dataset utilities
├── src/image_utils.py          # Image processing functions
├── src/feature_extraction.py   # Advanced feature computation
├── src/spatial_aware_algo.py   # Spatial assignment algorithm
├── src/mosaic_generator.py     # Mosaic assembly functions
├── src/visualization.py        # Result display utilities
├── src/utils.py               # General utility functions
├── requirements.txt       # Python dependencies
├── davinki.ipynb          # Jupyter notebook demonstration
├── img/              # Sample inputs and outputs
└── README.md              # This file
```


## Future Enhancements

- [ ] Advanced feature descriptors (HOG, SIFT, deep learning features) and similarity metrics
- [ ] GPU acceleration support


## License

This project is licensed under the MIT License. See LICENSE file for details.


## Citation

If you use this code in academic work, please cite:
```
@software{davinki_photomosaic,
  title={DaVinki: Photomosaic Generator using CIFAR-10 Dataset},
  author={[Parth]},
  year={2025},
  url={https://github.com/nihilisticneuralnet/DaVinki}
}
```
