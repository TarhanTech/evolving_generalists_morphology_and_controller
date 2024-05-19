import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from deap import base, creator, tools, algorithms
from individual import *

def main():
    ind: Individual = Individual()
    ind.evaluate_fitness()

if __name__ == "__main__": main()