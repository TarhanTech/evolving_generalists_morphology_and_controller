import numpy as np

def generate_terrain(height, width, block_size):
    # Initialize the array with zeros
    terrain = np.zeros((height, width), dtype=int)
    
    # Determine the number of blocks in each dimension
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size
    
    # Iterate through each block and decide to fill with 1's based on a random choice
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            # Randomly decide if this block should be raised (1) or not (0)
            if np.random.rand() > 0.5:
                terrain[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1
                
    return terrain

# Generate a terrain of size 256x50 with block size 4
terrain_map = generate_terrain(1600, 200, 2)

terrain_map_str = "\n".join(" ".join(str(cell) for cell in row) for row in terrain_map)

file_path_xml = "./xml_models/test.xml"

xml_str = ""
with open(file_path_xml, 'r') as file:
    xml_str = file.read()
    xml_str = xml_str.replace("{{height_map}}", terrain_map_str)

generated_ant_xml = f"./generated_ant_xml_test.xml"
with open(generated_ant_xml, 'w') as file:
    file.write(xml_str)