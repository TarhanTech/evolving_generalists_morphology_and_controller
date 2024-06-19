from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from neural_network import NeuralNetwork
from mj_env import MJEnv
import pprint
import os
import math
from PIL import Image

import torch
from torch import Tensor
import torch.nn as nn

class Individual:
    def __init__(self, morph_params = None, nn_params = None):
        self.mjEnv = MJEnv(morph_params)
        self.controller = NeuralNetwork(nn_params).to("cuda")
        self.params_size =  self.mjEnv.morphology.total_params + self.controller.total_weigths

    def setup(self, params: Tensor):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup(morph_params, "hills")

    def evaluate_fitness(self):
        generated_ant_xml = f"./generated_ant_xml_{id(self)}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.morphology.modified_xml_str)

        env: AntEnv = gym.make("Ant-v4", xml_file=generated_ant_xml, terminate_when_unhealthy=False)

        total_reward = 0
        episodes: int = 1
        for _ in range(episodes):
            obs, _ = env.reset()
            prev_distance_from_origin: int = 0
            distance_counter: int = 0
            done: bool = False
            while not done:
                obs_tensor: Tensor = torch.from_numpy(obs).to("cuda")
                action = self.controller(obs_tensor)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if math.isclose(info["distance_from_origin"], prev_distance_from_origin, abs_tol=1e-2):
                    distance_counter += 1
                else:
                    distance_counter = 0
                prev_distance_from_origin = info["distance_from_origin"]
                done = (terminated or truncated or distance_counter > 100)
            env.close()
        
        if os.path.exists(generated_ant_xml):
                os.remove(generated_ant_xml)
        return total_reward / episodes
    
    def evaluate_fitness_rendered(self):
        generated_ant_xml = f"./generated_ant_xml_{id(self)}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.morphology.modified_xml_str)

        env: AntEnv = gym.make("Ant-v4", render_mode="human", xml_file=generated_ant_xml, terminate_when_unhealthy=False)
        self._print_env_info(env)

        obs, _ = env.reset()
        prev_distance_from_origin = 0
        distance_counter = 0
        total_reward = 0
        done = False
        while not done:
            obs_tensor: Tensor = torch.from_numpy(obs).to("cuda")
            action = self.controller(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            if math.isclose(info["distance_from_origin"], prev_distance_from_origin, abs_tol=1e-2):
                distance_counter += 1
            else:
                distance_counter = 0
            # print(f"prev_distance_from_origin: {prev_distance_from_origin}")
            # print(f"info['distance_from_origin']: {info['distance_from_origin']}")
            prev_distance_from_origin = info["distance_from_origin"]
            done = (terminated or truncated or distance_counter > 100)
        env.close()

        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)
        print(f"Total Reward: {total_reward}")
        return total_reward

    def make_screenshot(self, path: str):
        generated_ant_xml = f"./generated_ant_xml_{id(self)}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.morphology.modified_xml_str)

        env: AntEnv = gym.make("Ant-v4", render_mode="rgb_array", xml_file=generated_ant_xml, healthy_z_range=(0.26, 4), camera_name="track_1")

        env.reset()
        frame = env.render()
        image = Image.fromarray(frame)
        image.save(f"{path}")

        env.close
        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)
            
    def _print_env_info(self, env: AntEnv):
        print(f"Action Space:\n{env.action_space}\n")
        print(f"Observation Space:\n{env.observation_space}\n")

    def print_morph_info(self):
        print("Morphology Parameters:")
        pprint.pprint(self.morphology.morph_params_map)
        print("\n")

    def print_controller_info(self):
        print("Controller Parameters:")
        print(f"{self.controller.input_size} Inp (+1 bias) -> {self.controller.hidden_size} Hid (+1 bias) -> {self.controller.output_size} Out")
        print(f"Total Weights: {self.controller.total_weigths}")
