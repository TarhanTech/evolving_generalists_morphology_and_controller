import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv

xml_file = "./xml_models/test.xml"
env: AntEnv = gym.make("Ant-v4", render_mode="human", xml_file=xml_file)
env.reset()

for _ in range(500):
    action = env.action_space.sample()
    env.step(action)
    env.render()
env.close()