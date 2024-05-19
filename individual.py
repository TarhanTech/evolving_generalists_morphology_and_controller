import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from individual import *
import pprint

class Individual:
    def __init__(self):
        self.controller = NeuralNetwork()
        self.morphology = Morphology()
    
    def evaluate_fitness(self):
        env: AntEnv = gym.make("Ant-v4", render_mode="human", xml_file=self.morphology.ant_xml_path)
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            if terminated or truncated: break
        env.close()
        print(f"Total Reward: {total_reward}")
        return total_reward
    
    def print_info(self, env: AntEnv):
        print("Morphology Parameters")
        pprint.pprint(self.morphology.morph_params)
        print("\n")
        print(f"Action Space:\n{env.action_space}\n")
        print(f"Observation Space:\n{env.observation_space}\n")
    
class Morphology:
    def __init__(self):
        self.leg_length_range = (0.3, 1.5) 
        self.leg_width_range = (0.05, 0.5)

        self.np_morph_params = self.generate_random_morph_params()
        self.morph_params = {
            "aux_1_length": self.np_morph_params[0],
            "ankle_1_length": self.np_morph_params[1],
            "aux_2_length": self.np_morph_params[2],
            "ankle_2_length": self.np_morph_params[3],
            "aux_3_length": self.np_morph_params[4],
            "ankle_3_length": self.np_morph_params[5],
            "aux_4_length": self.np_morph_params[6],
            "ankle_4_length": self.np_morph_params[7],

            "aux_1_width": self.np_morph_params[8],
            "ankle_1_width": self.np_morph_params[9],
            "aux_2_width": self.np_morph_params[10],
            "ankle_2_width": self.np_morph_params[11],
            "aux_3_width": self.np_morph_params[12],
            "ankle_3_width": self.np_morph_params[13],
            "aux_4_width": self.np_morph_params[14],
            "ankle_4_width": self.np_morph_params[15],
        }
        self.ant_xml_path = "./xml_models/ant_evolved.xml"
        self.modified_xml_str = self.load_and_modify_xml("./xml_models/ant_with_keys.xml", self.ant_xml_path)

    def generate_random_morph_params(self):
        random_leg_lengths = np.random.uniform(self.leg_length_range[0], self.leg_length_range[1], 8)
        random_leg_widths = np.random.uniform(self.leg_width_range[0], self.leg_width_range[1], 8)

        return np.hstack((random_leg_lengths, random_leg_widths))
    
    def load_and_modify_xml(self, file_path, output_file_path):
        with open(file_path, 'r') as file:
            xml_str = file.read()

        for key, value in self.morph_params.items():
            xml_str = xml_str.replace(f'{{{key}}}', str(value))

        with open(output_file_path, 'w') as file:
            file.write(xml_str)

        return xml_str

class NeuralNetwork:
    def __init__(self):
        pass