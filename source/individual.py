import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from source.neural_network import NeuralNetwork
from source.mj_env import *
from source.globals import *
import os
import math
from PIL import Image
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

class Individual:

    def __init__(self, id, morph_params = None, nn_params = None):
        self.id = id
        self.mjEnv = MJEnv(id=id, morph_params=morph_params)
        self.controller = NeuralNetwork(id=id, nn_params=nn_params).to("cuda")
        self.params_size =  self.mjEnv.morphology.total_params + self.controller.total_weigths
        self.generation: int = 1
    
    def increment_generation(self):
        self.generation += 1

    def setup_ant_hills(self, params: Tensor, floor_height: float, scale: int):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_hills(morph_params, floor_height, scale)

    def setup_ant_rough(self, params: Tensor, floor_height: float, block_size: int):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_rough(morph_params, floor_height, block_size)

    def setup_ant_default(self, params: Tensor):
        nn_params, morph_params = torch.split(params, (self.controller.total_weigths, self.mjEnv.morphology.total_params))
        self.controller.set_nn_params(nn_params)
        self.mjEnv.setup_ant_default(morph_params=morph_params)

    def evaluate_fitness(self, render_mode: str = None) -> float:
        if self.mjEnv.has_invalid_parameters(): 
            return -self._penalty_function(penalty_scale_factor_err)

        generated_ant_xml: str = f"./{train_ant_xml_folder}/generated_ant_xml_{self.id}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.xml_str)

        env: AntEnv = gym.make("Ant-v4", xml_file=generated_ant_xml, healthy_z_range=(-1, 7.5), render_mode=render_mode)
        if render_mode == "rgb_array":
            env = RecordVideo(env, video_folder="./ant_vid", name_prefix="eval", episode_trigger=lambda x: True)
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
        return (total_reward / episodes) - self._penalty_function(penalty_scale_factor)
    
    def make_screenshot_ant(self, path: str):
        if self.mjEnv.has_invalid_parameters(): return
        
        generated_ant_xml = f"./{train_ant_xml_folder}/generated_ant_xml_{self.id}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.xml_str)

        env: AntEnv = gym.make("Ant-v4", render_mode="rgb_array", xml_file=generated_ant_xml, healthy_z_range=(-1, 7.5), camera_name="topdown")

        env.reset()
        frame = env.render()
        image = Image.fromarray(frame)
        image.save(f"{path}")

        env.close
        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)
    
    def make_screenshot_env(self, path: str):
        if self.mjEnv.has_invalid_parameters(): return
        
        generated_ant_xml = f"./{train_ant_xml_folder}/generated_ant_xml_{self.id}.xml"
        with open(generated_ant_xml, 'w') as file:
            file.write(self.mjEnv.xml_str)

        env: AntEnv = gym.make("Ant-v4", render_mode="rgb_array", xml_file=generated_ant_xml, healthy_z_range=(0.26, 4), camera_name="env")

        env.reset()
        frame = env.render()
        image = Image.fromarray(frame)
        image.save(f"{path}")

        env.close
        if os.path.exists(generated_ant_xml):
            os.remove(generated_ant_xml)

    def print_controller_info(self):
        print("Controller Parameters:")
        print(f"{self.controller.input_size} Inp (+1 bias) -> {self.controller.hidden_size} Hid (+1 bias) -> {self.controller.output_size} Out")
        print(f"Total Weights: {self.controller.total_weigths}")

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

    def _penalty_function(self, scalar: int) -> float:
        morph_params: Tensor = self.mjEnv.morphology.morph_params_tensor
        length_params, width_params = torch.split(morph_params, (Morphology.total_leg_length_params, Morphology.total_leg_width_params))

        sum_leg_length_diff = self._get_total_difference(length_params, Morphology.leg_length_range)
        sum_leg_width_diff = self._get_total_difference(width_params, Morphology.leg_width_range)

        penalty = scalar * (sum_leg_length_diff + sum_leg_width_diff) * (penalty_growth_rate**self.generation)
        return penalty 

    def _get_total_difference(self, params: Tensor, range) -> float:
        leg_length_lower_diff: Tensor = range[0] - params[params < range[0]]
        leg_length_upper_diff: Tensor = params[params > range[1]] - range[1]
        sum_leg_length_diff: float = leg_length_lower_diff.sum().item() + leg_length_upper_diff.sum().item()

        return sum_leg_length_diff
    
