import numpy as np
from PIL import Image
from noise import pnoise2

# Function to generate perlin noise
def generate_perlin_noise(width, height, scale=10, octaves=6, persistence=0.5, lacunarity=2.0):
    noise_array = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            noise_array[i][j] = pnoise2(i / scale, j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=width,
                                        repeaty=height,
                                        base=0)
    return noise_array

# Settings for the noise
width = 512  # Image width
height = 100 # Image height
scale = 1.00001  # Scale affects the 'zoom' of the noise
octaves = 6
persistence = 0.5
lacunarity = 2.0

noise_img = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity)

print(np.min(noise_img))
print(np.max(noise_img))

# Normalize to [0, 255] for image saving
normalized_img = np.floor(255 * (noise_img - np.min(noise_img)) / (np.max(noise_img) - np.min(noise_img))).astype(np.uint8)
normalized_img = (normalized_img > 127).astype(np.uint8) * 255

# Create and save the image using Pillow
image = Image.fromarray(normalized_img, mode='L')  # 'L' mode is for grayscale
image.save(f"./terrain_noise/terrain_test.png")