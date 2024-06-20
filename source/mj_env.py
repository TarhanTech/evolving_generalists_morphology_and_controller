import torch
from torch import Tensor
import numpy as np
from typing import Tuple
import random 

from PIL import Image
from noise import pnoise2

class MJEnv:
    def __init__(self, morph_params: Tensor = None):
        self.terrain_env = TerrainEnv()
        self.morphology = Morphology(morph_params)

        self.file_path_template_hills = "./xml_models/ant_hills_with_keys.xml"
        self.file_path_template_rough = "./xml_models/ant_with_keys.xml"
        self.xml_str = ""

    def setup(self, morph_params: Tensor, terrain_env: str):
        self.morphology.set_morph_params(morph_params)
        if terrain_env == "hills":
            self.set_xml_str_with_hills_terrain()
        elif terrain_env == "rough":
            self.set_xml_str_with_rough_terrain()
        else:
            assert False, f"Unsupported environment: {terrain_env}"

    def set_xml_str_with_hills_terrain(self):
        with open(self.file_path_template_hills, 'r') as file:
            temp_xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))

        self.terrain_env.setup_hills()
        for key, value in self.terrain_env.hills_params.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def set_xml_str_with_rough_terrain(self):
        with open(self.file_path_template_rough, 'r') as file:
            temp_xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str


class TerrainEnv:
    def __init__(self):
        self.hills_params = {
            "terrain_noise": f"./terrain_noise/generated_terrain_hills_{id(self)}.png",
            "floor_width": 150,
            "floor_length": 10,
            "floor_height": 1,
            "torso_pos": None
        }
        self.hills_params["torso_pos"] = f"{self.hills_params['floor_width'] - 5} 0 3"
        self.rough_params = {
            "hfield_ncol": 1500,
            "hfield_nrow": 100,
            "hfield_elevation": None,
            "floor_width": 150,
            "floor_length": 10
        }

    def setup_hills(self):
        width: int = 300
        height: int = 20
        noise_image = self._generate_noise_image(width, height)

        normalized_img = np.floor(255 * (noise_image - np.min(noise_image)) / (np.max(noise_image) - np.min(noise_image))).astype(np.uint8)
        # normalized_img = (normalized_img > 127).astype(np.uint8) * 255

        image = Image.fromarray(normalized_img, mode="L")  # 'L' mode is for grayscale
        image.save(f"./terrain_noise/generated_terrain_hills_{id(self)}.png")


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
        return noise_array
    
    def setup_rough(self):
        pass


class Morphology:
    def __init__(self, morph_params: Tensor = None):
        self.leg_length_range = (0.3, 1.5)
        self.initial_leg_length_range_size = 1.2

        # self.leg_width_range = (0.05, 0.5)
        
        self.total_params = 8 # If using also width of the legs then use 16
        self.morph_params_tensor: Tensor = None
        self.morph_params_map = None 


        if(morph_params is None):
            self.set_morph_params(self.generate_random_morph_params())
        else:
            self.set_morph_params(morph_params)

    def generate_random_morph_params(self):
        random_leg_lengths = torch.FloatTensor(8).uniform_(self.leg_length_range[0], self.leg_length_range[1])
        # random_leg_widths = torch.FloatTensor(8).uniform_(self.leg_width_range[0], self.leg_width_range[1])

        return random_leg_lengths # torch.cat((random_leg_lengths, random_leg_widths), dim=0)

    def set_morph_params(self, morph_params: Tensor):
        assert morph_params.size(0) == self.total_params, (
            f"Expected {self.total_params} parameters, but got {morph_params.size(0)}."
        )

        self.morph_params_tensor = morph_params.clone()
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

    def generate_initial_leg_length_range(self) -> Tuple[float, float]:
        assert self.initial_leg_length_range_size <= (self.leg_length_range[1] - self.leg_length_range[0]), (
            f"The sample range {self.initial_leg_length_range_size} is bigger then the range it samples from."
        )
        min_start: float = self.leg_length_range[0]
        max_start: float = self.leg_length_range[1] - self.initial_leg_length_range_size

        start: float = random.uniform(min_start, max_start)
        end: float = start + self.initial_leg_length_range_size
        print(f"Generated bounds for the leg lengths are ({start}, {end})")

        return (start, end)
