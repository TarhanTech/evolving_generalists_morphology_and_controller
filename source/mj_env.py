"""Module containing the Mujoco definitions."""

import math
import uuid
import os
import torch
from torch import Tensor
import numpy as np

from PIL import Image
from noise import pnoise2


class MJEnv:
    """Class defining the mujoco environment, so the agent and environment."""

    def __init__(self, uid: uuid.UUID, morph_params_bounds_enc: tuple[float, float], dis_morph_evo: bool):
        self.dis_morph_evo = dis_morph_evo
        self.uid: uuid.UUID = uid
        self.terrain_env = TerrainEnv(uid)
        self.morphology = Morphology(uid, morph_params_bounds_enc, dis_morph_evo)

        self.file_path_template_hills = "./xml_models/ant_hills_with_keys.xml"
        self.file_path_template_rough = "./xml_models/ant_rough_with_keys.xml"
        self.file_path_template_default = "./xml_models/ant_plain_with_keys.xml"
        self.xml_str = ""

    def setup_ant_hills(self, floor_height: float, scale: int, morph_params: Tensor = None):
        if self.dis_morph_evo is False and morph_params is None:
            raise Exception(f"Morphological evolution is activated, so morph_params is expected to have a value.")
    
        self.terrain_env.setup_hills(floor_height, scale)

        if self.dis_morph_evo is False:
            self.morphology.set_morph_params(morph_params)
            
        temp_xml_str = self._create_xml_with_morph_params(
            template_xml=self.file_path_template_hills
        )
        for key, value in self.terrain_env.hills_params.items():
            temp_xml_str = temp_xml_str.replace(f"{{{key}}}", str(value))
        self.xml_str = temp_xml_str

    def setup_ant_rough(self, floor_height: float, block_size: int, morph_params: Tensor = None):
        if self.dis_morph_evo is False and morph_params is None:
            raise Exception(f"Morphological evolution is activated, so morph_params is expected to have a value.")
        
        self.terrain_env.setup_rough(floor_height, block_size)

        if self.dis_morph_evo is False:
            self.morphology.set_morph_params(morph_params)
        
        temp_xml_str = self._create_xml_with_morph_params(
            template_xml=self.file_path_template_rough
        )

        for key, value in self.terrain_env.rough_params.items():
            temp_xml_str = temp_xml_str.replace(f"{{{key}}}", str(value))

        self.xml_str = temp_xml_str

    def setup_ant_default(self, morph_params: Tensor = None):
        if self.dis_morph_evo is False and morph_params is None:
            raise Exception(f"Morphological evolution is activated, so morph_params is expected to have a value.")
        
        if self.dis_morph_evo is False:
            self.morphology.set_morph_params(morph_params)

        self.xml_str = self._create_xml_with_morph_params(
            template_xml=self.file_path_template_default
        )

    def has_invalid_parameters(self) -> bool:
        return any(
            param < 0.0001 for param in self.morphology.morph_params_map.values()
        )  # mujoco does not allow numbers lower then 1e-15

    def _create_xml_with_morph_params(self, template_xml: str):
        with open(template_xml, "r") as file:
            xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            xml_str = xml_str.replace(f"{{{key}}}", str(value))

        return xml_str


class TerrainEnv:
    def __init__(self, uid: uuid.UUID):
        self.uid: uuid.UUID = uid

        # The difficulty increase comes from the floor_height param
        self.hills_params = {
            "terrain_noise": None,
            "floor_width": 150,
            "floor_length": 80,
            "floor_height": 1,
            "floor_pos": None,
        }
        self.hills_params["floor_pos"] = f"{self.hills_params['floor_width'] - 5} 0 0"

        # The difficulty increase comes from the floor_height param
        self.rough_params = {
            "hfield_ncol": None,
            "hfield_nrow": None,
            "hfield_elevation": None,
            "floor_width": 150,
            "floor_length": 80,
            "floor_height": 0.1,
            "floor_pos": None,
        }
        self.rough_params["hfield_ncol"] = self.rough_params["floor_width"] * 5
        self.rough_params["hfield_nrow"] = self.rough_params["floor_length"] * 5
        self.rough_params["floor_pos"] = f"{self.rough_params['floor_width'] - 5} 0 0"

    def setup_hills(self, floor_height: float, scale: int):
        self.hills_params["floor_height"] = floor_height
        # if self.hills_params["terrain_noise"] is not None and os.path.exists(self.hills_params["terrain_noise"]):
        #     os.remove(self.hills_params["terrain_noise"])

        width: int = self.hills_params["floor_width"] * 2
        height: int = self.hills_params["floor_length"] * 2
        noise_image = self._generate_noise_image(width, height, scale)

        image = Image.fromarray(noise_image, mode="L")  # 'L' mode is for grayscale
        terrain_noise_file = f"./train_terrain_noise/generated_terrain_hills_{self.uid}.png"
        image.save(terrain_noise_file)
        self.hills_params["terrain_noise"] = os.path.abspath(terrain_noise_file)

    def _generate_noise_image(
        self, width, height, scale, octaves=6, persistence=0.5, lacunarity=2.0
    ):
        noise_array = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                noise_array[i][j] = pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=width,
                    repeaty=height,
                    base=0,
                )

        normalized_noise_image = np.floor(
            255 * (noise_array - np.min(noise_array)) / (np.max(noise_array) - np.min(noise_array))
        ).astype(np.uint8)
        # normalized_noise_image = (normalized_img > 127).astype(np.uint8) * 255
        return normalized_noise_image

    def setup_rough(self, floor_height: float, block_size: int):
        self.rough_params["floor_height"] = floor_height

        rough_terrain_str: str = self._generate_rough_terrain(
            self.rough_params["hfield_nrow"],
            self.rough_params["hfield_ncol"],
            floor_height,
            block_size,
        )
        self.rough_params["hfield_elevation"] = rough_terrain_str

    def _generate_rough_terrain(self, row: int, col: int, floor_height: float, block_size: int):
        terrain = np.zeros((row, col), dtype=int)

        num_blocks_vertical = row // block_size
        num_blocks_horizontal = col // block_size
        num_heights: int = math.ceil((floor_height / 0.1) + 1)

        for i in range(num_blocks_vertical):
            for j in range(num_blocks_horizontal):
                # Randomly decide if this block should be raised (1) or not (0)
                terrain[
                    i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size
                ] = np.random.randint(0, num_heights)

        rough_terrain_str = " ".join(" ".join(str(cell) for cell in row) for row in terrain)
        return rough_terrain_str


class Morphology:
    leg_length_range = (0.1, 1.5)
    total_leg_length_params: int = 8

    leg_width_range = (0.05, 0.2)
    total_leg_width_params: int = 8

    def __init__(self, uid: uuid.UUID, morph_params_bounds_enc: tuple[float, float], dis_morph_evo: bool):
        self.dis_morp_evo = dis_morph_evo
        self.total_params: int = 0 if self.dis_morp_evo else self.total_leg_length_params + self.total_leg_width_params

        self.uid: uuid.UUID = uid
        self.morph_params_bounds_enc: tuple[float, float] = morph_params_bounds_enc
        self.morph_params_tensor: Tensor = None
        self.morph_params_map = self._get_default_morph_params_map()

    def set_morph_params(self, morph_params: Tensor):
        if self.dis_morp_evo == True: raise Exception(f"Morphological evolution is disabled. Setting custom morph parameters is not supported.")

        assert (
            morph_params.size(0) == self.total_params
        ), f"Expected {self.total_params} parameters, but got {morph_params.size(0)}."

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
        assert (
            len(self.morph_params_map) == self.total_params
        ), f"Expected self.morph_params_map to have {self.total_params} elements, but has {len(self.morph_params_map)}."

    def _get_default_morph_params_map(self):
        return {
            "aux_1_length": 0.2,
            "ankle_1_length": 0.4,
            "aux_2_length": 0.2,
            "ankle_2_length": 0.4,
            "aux_3_length": 0.2,
            "ankle_3_length": 0.4,
            "aux_4_length": 0.2,
            "ankle_4_length": 0.4,
            "aux_1_width": 0.08,
            "ankle_1_width": 0.08,
            "aux_2_width": 0.08,
            "ankle_2_width": 0.08,
            "aux_3_width": 0.08,
            "ankle_3_width": 0.08,
            "aux_4_width": 0.08,
            "ankle_4_width": 0.08,
        }

    def _decode_morph_params(self, morph_params: Tensor) -> Tensor:
        length_params, width_params = torch.split(
            morph_params, (self.total_leg_length_params, self.total_leg_width_params)
        )

        a: float = (self.leg_length_range[1] - self.leg_length_range[0]) / (
            self.morph_params_bounds_enc[1] - self.morph_params_bounds_enc[0]
        )
        b: float = self.leg_length_range[0] - (a * self.morph_params_bounds_enc[0])
        decoded_leg_length: Tensor = (a * length_params) + b

        c: float = (self.leg_width_range[1] - self.leg_width_range[0]) / (
            self.morph_params_bounds_enc[1] - self.morph_params_bounds_enc[0]
        )
        d: float = self.leg_width_range[0] - (c * self.morph_params_bounds_enc[0])
        decoded_leg_width: Tensor = (c * width_params) + d

        return torch.cat((decoded_leg_length, decoded_leg_width), dim=0)
