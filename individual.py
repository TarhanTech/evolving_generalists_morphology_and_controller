import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from individual import *
import pprint
import os

import torch
from torch import Tensor
import torch.nn as nn

class Individual:
    def __init__(self, morph_params = None, nn_params = None):
        self.morphology = Morphology(morph_params)
        self.controller = NeuralNetwork(nn_params).to("cuda")
        self.params_size =  self.morphology.total_genes + self.controller.total_weigths

    def evaluate_fitness(self):
        generated_ant_xml = f"./generated_ant_xml_{id(self)}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.morphology.modified_xml_str)

        env: AntEnv = gym.make("Ant-v4", xml_file=generated_ant_xml)

        obs, _ = env.reset()
        total_reward = 0
        while True:
            obs_tensor: Tensor = torch.from_numpy(obs).to("cuda")
            action = self.controller(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated: 
                break
        env.close()
        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)
        return total_reward
    
    def evaluate_fitness_rendered(self):
        generated_ant_xml = f"./generated_ant_xml_{id(self)}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.morphology.modified_xml_str)

        env: AntEnv = gym.make("Ant-v4", render_mode="human", xml_file=generated_ant_xml)
        self._print_env_info(env)

        obs, _ = env.reset()
        total_reward = 0
        while True:
            obs_tensor: Tensor = torch.from_numpy(obs).to("cuda")
            action = self.controller(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            if terminated or truncated: break
        env.close()
        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)
        print(f"Total Reward: {total_reward}")
        return total_reward
    
    def _print_env_info(self, env: AntEnv):
        print(f"Action Space:\n{env.action_space}\n")
        print(f"Observation Space:\n{env.observation_space}\n")

    def print_morph_info(self):
        print("Morphology Parameters:")
        pprint.pprint(self.morphology.morph_params_map)
        print("\n")

    def print_controller_info(self):
        print("Controller Parameters:")
        print(f"{self.controller.input_size} Inp -> {self.controller.hidden_size} Hid -> {self.controller.output_size} Out")
        print(f"Total Weights: {self.controller.total_weigths}")

    
class Morphology:
    def __init__(self, morph_params: Tensor = None):
        self.leg_length_range = (0.3, 1.5) 
        self.leg_width_range = (0.05, 0.5)
        self.total_genes = 16

        self.morph_params_tensor: Tensor = None
        self.morph_params_map = None 

        self.modified_xml_str = ""

        if(morph_params is None):
            self.set_morph_params(self.generate_random_morph_params())
        else:
            self.set_morph_params(morph_params)

    def generate_random_morph_params(self):
        random_leg_lengths = torch.FloatTensor(8).uniform_(self.leg_length_range[0], self.leg_length_range[1])
        random_leg_widths = torch.FloatTensor(8).uniform_(self.leg_width_range[0], self.leg_width_range[1])

        return torch.cat((random_leg_lengths, random_leg_widths), dim=0)
    
    def create_xml_str(self):
        file_path_ant_with_keys = "./xml_models/ant_with_keys.xml"

        with open(file_path_ant_with_keys, 'r') as file:
            xml_str = file.read()

        for key, value in self.morph_params_map.items():
            xml_str = xml_str.replace(f'{{{key}}}', str(value))

        return xml_str

    def set_morph_params(self, morph_params: Tensor):
        assert morph_params.size(0) == self.total_genes, (
            f"Expected {self.total_genes} parameters, but got {morph_params.size(0)}."
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

            "aux_1_width": self.morph_params_tensor[8].item(),
            "ankle_1_width": self.morph_params_tensor[9].item(),
            "aux_2_width": self.morph_params_tensor[10].item(),
            "ankle_2_width": self.morph_params_tensor[11].item(),
            "aux_3_width": self.morph_params_tensor[12].item(),
            "ankle_3_width": self.morph_params_tensor[13].item(),
            "aux_4_width": self.morph_params_tensor[14].item(),
            "ankle_4_width": self.morph_params_tensor[15].item(),
        } 

        self.modified_xml_str = self.create_xml_str()


class NeuralNetwork(nn.Module):
    def __init__(self, nn_params: Tensor = None):
        super(NeuralNetwork, self).__init__()

        self.bias_size: int = 1
        self.input_size: int = 27
        self.hidden_size: int = 20
        self.output_size: int = 8
        self.total_weigths: int = ((self.input_size + self.bias_size) * self.hidden_size) + ((self.hidden_size + self.bias_size) * self.output_size)

        self.fc1: nn.Linear = nn.Linear(self.input_size, self.hidden_size).double()
        self.fc2: nn.Linear = nn.Linear(self.hidden_size, self.output_size).double()
        self.tanh: nn.Tanh = nn.Tanh()

        if nn_params is not None:
            self.set_nn_params(nn_params)

    def forward(self, x: Tensor) -> np.ndarray:
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x.to("cpu").detach().numpy()
    
    def set_nn_params(self, nn_params: Tensor):
        assert nn_params.size(0) == self.total_weigths, (
            f"Expected {self.total_weigths} parameters, but got {nn_params.size(0)}."
        )

        input_weight_size = self.input_size * self.hidden_size
        input_bias_size = self.hidden_size

        hidden_weight_size = self.hidden_size * self.output_size

        self.fc1.weight = nn.Parameter(nn_params[:input_weight_size].view(self.hidden_size, self.input_size).clone())
        self.fc1.bias = nn.Parameter(nn_params[input_weight_size:input_weight_size + input_bias_size].clone())

        self.fc2.weight = nn.Parameter(nn_params[input_weight_size + input_bias_size:input_weight_size + input_bias_size + hidden_weight_size].view(self.output_size, self.hidden_size).clone())
        self.fc2.bias = nn.Parameter(nn_params[input_weight_size + input_bias_size + hidden_weight_size:].clone())
