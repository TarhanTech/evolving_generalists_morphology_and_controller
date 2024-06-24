import torch
from torch import Tensor
import numpy as np
import os

from source.globals import *
from PIL import Image
from noise import pnoise2

class MJEnv:
    def __init__(self, id, morph_params: Tensor = None):
        self.id = id
        self.terrain_env = TerrainEnv(id=id)
        self.morphology = Morphology(id=id, morph_params=morph_params)

        self.file_path_template_hills = "./xml_models/ant_hills_with_keys.xml"
        self.file_path_template_rough = "./xml_models/ant_rough_with_keys.xml"
        self.file_path_template_default = "./xml_models/ant_default_with_keys.xml" # TODO: create this xml file with the keys
        self.xml_str = ""

    def setup_ant_hills(self, morph_params: Tensor, floor_height: float):
        self.morphology.set_morph_params(morph_params)
        temp_xml_str = self._create_xml_with_morph_params(template_xml=self.file_path_template_hills)

        self.terrain_env.setup_hills(floor_height)
        for key, value in self.terrain_env.hills_params.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def setup_ant_rough(self, morph_params: Tensor, floor_height: float):
        self.morphology.set_morph_params(morph_params)
        temp_xml_str = self._create_xml_with_morph_params(template_xml=self.file_path_template_rough)

        self.terrain_env.setup_rough(floor_height)
        for key, value in self.terrain_env.rough_params.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def setup_ant_default(self, morph_params: Tensor):
        self.morphology.set_morph_params(morph_params)
        self.xml_str = self._create_xml_with_morph_params() # TODO: pass in the template for default env

    def has_invalid_parameters(self) -> bool:
        return any(param < 0 for param in self.morphology.morph_params_map.values())

    def _create_xml_with_morph_params(self, template_xml: str):
        with open(template_xml, 'r') as file:
            xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            xml_str = xml_str.replace(f'{{{key}}}', str(value))

        return xml_str

class TerrainEnv:
    def __init__(self, id):
        self.id = id

        # The difficulty increase comes from the floor_height param
        self.hills_params = {
            "terrain_noise": None,
            "floor_width": 150,
            "floor_length": 20,
            "floor_height": 1,
            "floor_pos": None
        }
        self.hills_params["floor_pos"] = f"{self.hills_params['floor_width'] - 5} 0 0"
        
        # The difficulty increase comes from the floor_height param
        self.rough_params = {
            "hfield_ncol": 1500,
            "hfield_nrow": 100,
            "hfield_elevation": None,
            "floor_width": 150,
            "floor_length": 20,
            "floor_heigth": 0.1,
            "floor_pos": None
        }
        self.rough_params["floor_pos"] = f"{self.rough_params['floor_width'] - 5} 0 0"

    def setup_hills(self, floor_height: float):
        self.hills_params["floor_height"] = floor_height
        if self.hills_params["terrain_noise"] is not None and os.path.exists(self.hills_params["terrain_noise"]):
            os.remove(self.hills_params["terrain_noise"])

        width: int = self.hills_params["floor_width"] * 2
        height: int = self.hills_params["floor_length"] * 2
        noise_image = self._generate_noise_image(width, height)

        image = Image.fromarray(noise_image, mode="L")  # 'L' mode is for grayscale
        terrain_noise_file = f"./train_terrain_noise/generated_terrain_hills_{self.id}.png"
        image.save(terrain_noise_file)
        self.hills_params["terrain_noise"] = os.path.abspath(terrain_noise_file)

    def _generate_noise_image(self, width, height, scale=5, octaves=6, persistence=0.5, lacunarity=2.0):
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
                
        normalized_noise_image = np.floor(255 * (noise_array - np.min(noise_array)) / (np.max(noise_array) - np.min(noise_array))).astype(np.uint8)
        # normalized_noise_image = (normalized_img > 127).astype(np.uint8) * 255
        return normalized_noise_image
    
    def setup_rough(self, floor_heigth: float):
        self.rough_params["floor_heigth"] = floor_heigth
        # TODO: Make the rough terrain generation based on the floor height. 

        rough_terrain_str: str = self._generate_rough_terrain(self.rough_params["hfield_nrow"], self.rough_params["hfield_ncol"], 1)
        self.rough_params["hfield_elevation"] = rough_terrain_str

    def _generate_rough_terrain(self, row, col, block_size):
        terrain = np.zeros((row, col), dtype=int)
        
        num_blocks_vertical = row // block_size
        num_blocks_horizontal = col // block_size
        
        for i in range(num_blocks_vertical):
            for j in range(num_blocks_horizontal):
                # Randomly decide if this block should be raised (1) or not (0)
                if np.random.rand() > 0.5:
                    terrain[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1

        rough_terrain_str = " ".join(" ".join(str(cell) for cell in row) for row in terrain)            
        return rough_terrain_str

class Morphology:
    leg_length_range = (0.1, 1.5)
    leg_width_range = (0.05, 0.5)
    total_params = 8 # TODO: If using also width of the legs then use 16
    
    def __init__(self, id, morph_params: Tensor = None):
        self.id = id
        
        self.morph_params_tensor: Tensor = morph_params
        self.morph_params_map = None 

    def set_morph_params(self, morph_params: Tensor):
        assert morph_params.size(0) == self.total_params, (
            f"Expected {self.total_params} parameters, but got {morph_params.size(0)}."
        )

        self.morph_params_tensor = self._decode_morph_params(morph_params.clone())
        self.morph_params_map = {
            "aux_1_length": self.morph_params_tensor[0].item(),
            "ankle_1_length": self.morph_params_tensor[1].item(),
            "aux_2_length": self.morph_params_tensor[2].item(),
            "ankle_2_length": self.morph_params_tensor[3].item(),
            "aux_3_length": self.morph_params_tensor[4].item(),
            "ankle_3_length": self.morph_params_tensor[5].item(),
            "aux_4_length": self.morph_params_tensor[6].item(),
            "ankle_4_length": self.morph_params_tensor[7].item(),

            # "aux_1_width": self.morph_params_tensor[8].item(),
            # "ankle_1_width": self.morph_params_tensor[9].item(),
            # "aux_2_width": self.morph_params_tensor[10].item(),
            # "ankle_2_width": self.morph_params_tensor[11].item(),
            # "aux_3_width": self.morph_params_tensor[12].item(),
            # "ankle_3_width": self.morph_params_tensor[13].item(),
            # "aux_4_width": self.morph_params_tensor[14].item(),
            # "ankle_4_width": self.morph_params_tensor[15].item(),
        }

        assert len(self.morph_params_map) == self.total_params, (
            f"Expected self.morph_params_map to have {self.total_params} elements, but has {len(self.morph_params_map)}."
        )
    
    def _decode_morph_params(self, morph_params: Tensor) -> Tensor:
        # TODO: If using width of length modify this to conform
        a = (self.leg_length_range[1] - self.leg_length_range[0]) / (algo_params_range[1] - algo_params_range[0])
        b = self.leg_length_range[0] - (a * algo_params_range[0])

        return (a * morph_params) + b
