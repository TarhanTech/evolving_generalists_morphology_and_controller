import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os

def main():
    xml_relative_path = "./models/ant.xml"
    xml_file_path = os.path.join(os.getcwd(), xml_relative_path)

    model = PPO.load("my_ppo_ant")
    eval_env: AntEnv = gym.make("Ant-v4", render_mode="human", xml_file=xml_file_path)

    obs, _ = eval_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        eval_env.render()
        if terminated or truncated:
            print(f"Total Reward: {total_reward}")
            break

if __name__ == "__main__": main()