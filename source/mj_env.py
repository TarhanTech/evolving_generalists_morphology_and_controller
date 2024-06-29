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
        self.file_path_template_default = "./xml_models/ant_plain_with_keys.xml"
        self.xml_str = ""

    def setup_ant_hills(self, morph_params: Tensor, floor_height: float):
        self.morphology.set_morph_params(morph_params)
        temp_xml_str = self._create_xml_with_morph_params(template_xml=self.file_path_template_hills)

        self.terrain_env.setup_hills(floor_height)
        for key, value in self.terrain_env.hills_params.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def setup_ant_rough(self, morph_params: Tensor, floor_height: float, block_size: int):
        self.morphology.set_morph_params(morph_params)
        temp_xml_str = self._create_xml_with_morph_params(template_xml=self.file_path_template_rough)

        self.terrain_env.setup_rough(floor_height, block_size)
        for key, value in self.terrain_env.rough_params.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def setup_ant_default(self, morph_params: Tensor):
        self.morphology.set_morph_params(morph_params)
        self.xml_str = self._create_xml_with_morph_params(template_xml=self.file_path_template_default)

    def has_invalid_parameters(self) -> bool:
        return any(param < 0.0001 for param in self.morphology.morph_params_map.values()) # mujoco does not allow numbers lower then 1e-15   

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
            "hfield_ncol": None,
            "hfield_nrow": None,
            "hfield_elevation": None,
            "floor_width": 150,
            "floor_length": 40,
            "floor_heigth": 0.1,
            "floor_pos": None
        }
        self.rough_params["hfield_ncol"] = self.rough_params["floor_width"] * 5
        self.rough_params["hfield_nrow"] = self.rough_params["floor_length"] * 5
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
    
    def setup_rough(self, floor_heigth: float, block_size: int):
        self.rough_params["floor_heigth"] = floor_heigth
        # TODO: Make the rough terrain generation based on the floor height. 

        rough_terrain_str: str = self._generate_rough_terrain(self.rough_params["hfield_nrow"], self.rough_params["hfield_ncol"], floor_heigth, block_size)
        self.rough_params["hfield_elevation"] = rough_terrain_str

    def _generate_rough_terrain(self, row: int, col: int, floor_height: float, block_size: int):
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
    total_leg_length_params: int = 8

    leg_width_range = (0.08, 0.2)
    total_leg_width_params: int = 8

    total_params: int = total_leg_length_params + total_leg_width_params
    
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

            "aux_1_width": self.morph_params_tensor[8].item(),
            "ankle_1_width": self.morph_params_tensor[9].item(),
            "aux_2_width": self.morph_params_tensor[10].item(),
            "ankle_2_width": self.morph_params_tensor[11].item(),
            "aux_3_width": self.morph_params_tensor[12].item(),
            "ankle_3_width": self.morph_params_tensor[13].item(),
            "aux_4_width": self.morph_params_tensor[14].item(),
            "ankle_4_width": self.morph_params_tensor[15].item(),
        }
        assert len(self.morph_params_map) == self.total_params, (
            f"Expected self.morph_params_map to have {self.total_params} elements, but has {len(self.morph_params_map)}."
        )
    
    def _decode_morph_params(self, morph_params: Tensor) -> Tensor:
        length_params, width_params = torch.split(morph_params, (self.total_leg_length_params, self.total_leg_width_params))

        a: float = (self.leg_length_range[1] - self.leg_length_range[0]) / (algo_params_range[1] - algo_params_range[0])
        b: float = self.leg_length_range[0] - (a * algo_params_range[0])
        decoded_leg_length: Tensor = (a * length_params) + b

        c: float = (self.leg_width_range[1] - self.leg_width_range[0]) / (algo_params_range[1] - algo_params_range[0])
        d: float = self.leg_width_range[0] - (c * algo_params_range[0])
        decoded_leg_width: Tensor = (c * width_params) + d

        return torch.cat((decoded_leg_length, decoded_leg_width), dim=0)
