import os
from PIL import Image

def save_results(results, output_dir='./'):
    os.makedirs(output_dir, exist_ok=True)
    mosaic_pil = Image.fromarray(results['mosaic'])
    mosaic_pil.save(os.path.join(output_dir, 'mosaic.png'))