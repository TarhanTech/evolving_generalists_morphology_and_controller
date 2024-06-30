from source.individual import *
from source.training_env import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tensor", type=str, help="Path to a tensor.pt file that should be tested")
args = parser.parse_args()
params = torch.load(args.tensor) 

ind: Individual = Individual(id=101)
tr_schedule: TrainingSchedule = TrainingSchedule()

folder_run_data: str = f"./env_images"
os.makedirs(folder_run_data, exist_ok=True)
os.makedirs(f"{folder_run_data}/rough", exist_ok=True)
os.makedirs(f"{folder_run_data}/hills", exist_ok=True)

for training_env in tr_schedule.training_schedule:
    if isinstance(training_env, RoughTerrain):
        ind.setup_ant_rough(params, training_env.floor_height, training_env.block_size)
        ind.make_screenshot_env(f"{folder_run_data}/rough/rough_{training_env.block_size}_{training_env.floor_height}.png")
    elif isinstance(training_env, HillsTerrain):
        ind.setup_ant_hills(params, training_env.floor_height, training_env.scale)
        ind.make_screenshot_env(f"{folder_run_data}/hills/hills_{training_env.scale}_{training_env.floor_height}.png")
    elif isinstance(training_env, DefaultTerrain):
        ind.setup_ant_default(params)
        ind.make_screenshot_env(f"{folder_run_data}/default_terrain.png")
    else:
        assert False, "Class type not supported"
