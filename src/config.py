class Config:
    def __init__(self, target_image_path, tile_size=32, grid_size=(64, 64)):
        self.target_image_path = target_image_path
        self.tile_size = tile_size
        self.grid_size = grid_size