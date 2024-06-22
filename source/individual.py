import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from source.neural_network import NeuralNetwork
from source.mj_env import MJEnv
import pprint
from source.globals import *
import os
import math
from PIL import Image
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation

class Individual:

    def __init__(self, id, morph_params = None, nn_params = None):
        self.id = id
        self.mjEnv = MJEnv(id=id, morph_params=morph_params)
        self.controller = NeuralNetwork(id=id, nn_params=nn_params).to("cuda")
        self.params_size =  self.mjEnv.morphology.total_params + self.controller.total_weigths
    
    def setup_ant_hills(self, params: Tensor, floor_height: float):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_hills(morph_params=morph_params, floor_height=floor_height)

    def setup_ant_rough(self, params: Tensor, floor_height: float):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_rough(morph_params=morph_params, floor_height=floor_height)

    def setup_ant_default(self, params: Tensor):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_default(morph_params=morph_params)

    def evaluate_fitness(self, render_mode: str = None):
        generated_ant_xml: str = f"./{train_ant_xml_folder}/generated_ant_xml_{self.id}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.xml_str)

        env: AntEnv = gym.make("Ant-v4", xml_file=generated_ant_xml, terminate_when_unhealthy=False, render_mode=render_mode)

        total_reward: float = 0
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
                if render_mode == "human": env.render()

                if math.isclose(info["distance_from_origin"], prev_distance_from_origin, abs_tol=1e-2):
                    distance_counter += 1
                else:
                    distance_counter = 0
                prev_distance_from_origin = info["distance_from_origin"]
                done = (terminated or truncated or distance_counter > 300 or self._is_upside_down(env))
            env.close()
        
        # if os.path.exists(generated_ant_xml):
        #     os.remove(generated_ant_xml)
        return total_reward / episodes

    def make_screenshot(self, path: str):
        generated_ant_xml = f"./{train_ant_xml_folder}/generated_ant_xml_{self.id}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.xml_str)

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

    def _is_upside_down(self, env: AntEnv):
        q = env.get_wrapper_attr("data").body("torso").xquat
        q = [q[1], q[2], q[3], q[0]] # convert quaternion from wxyz to xyzw
        q = Rotation.from_quat(q)

        rotation_matrix = q.as_matrix()
        up_vector = rotation_matrix[2] # z-vector

        return up_vector[2] < -0.75

    def print_mjenv_info(self):
        print("Morphology Parameters:")
        pprint.pprint(self.mjEnv.morphology.morph_params_map)
        print("\n")

    def print_controller_info(self):
        print("Controller Parameters:")
        print(f"{self.controller.input_size} Inp (+1 bias) -> {self.controller.hidden_size} Hid (+1 bias) -> {self.controller.output_size} Out")
        print(f"Total Weights: {self.controller.total_weigths}")
