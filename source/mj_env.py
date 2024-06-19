import torch
from torch import Tensor
import numpy as np
from typing import Tuple
import random 

class MJEnv:
    def __init__(self, morph_params: Tensor = None):
        self.env = TerrainEnv()
        self.morphology = Morphology(morph_params)

        self.file_path_template_hills = "./xml_models/ant_with_keys.xml"
        self.file_path_template_rough = "./xml_models/ant_with_keys.xml"
        self.xml_str = ""

    def setup(self, morph_params: Tensor, env: str):
        self.morphology.set_morph_params(morph_params)
        if env == "hills":
            self.set_xml_str_for_hills_terrain()
        elif env == "rough":
            self.set_xml_str_for_rough_terrain()
        else:
            assert False, f"Unsupported environment: {env}"

    def set_xml_str_for_hills_terrain(self):
        with open(self.file_path_template_hills, 'r') as file:
            temp_xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str

    def set_xml_str_for_rough_terrain(self):
        with open(self.file_path_template_rough, 'r') as file:
            temp_xml_str = file.read()

        for key, value in self.morphology.morph_params_map.items():
            temp_xml_str = temp_xml_str.replace(f'{{{key}}}', str(value))
        
        self.xml_str = temp_xml_str


class TerrainEnv:
    def __init(self):
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
