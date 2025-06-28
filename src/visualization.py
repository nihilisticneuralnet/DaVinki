import matplotlib.pyplot as plt

def display_results(target_image, results):
    fig, axes = plt.subplots(1,2, figsize=(15, 15))
    
    axes[0, 0].imshow(target_image)
    axes[0, 0].set_title('Target Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['mosaic'])
    axes[0, 1].set_title('Mosaic Image')
    axes[0, 1].axis('off')
            
    plt.tight_layout()
    plt.show()