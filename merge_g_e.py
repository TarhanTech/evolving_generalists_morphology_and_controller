from pathlib import Path
import torch
from torch import Tensor
from source.training_env import TrainingSchedule, TerrainType, HillsTerrain
import pickle

def load_pickle(file_path: Path):
    with open(file_path, "rb") as file:
        v = pickle.load(file)
    return v

def save_pickle(filename: str, data: any, save_path: Path):
    """Saves data to a pickle"""
    pickle_path: Path = save_path / filename
    with open(pickle_path, "wb") as file:
        pickle.dump(data, file)

run_path_1: Path = Path("runs/Spec-MorphEvo-Long/Spec-MorphEvo-Long_1719_01-06_15-35-23-123456")
# run_path_2: Path = Path("runs/Spec-MorphEvo-Long_30-11_13-38-33-376480")

g_1 = load_pickle(run_path_1 / "G_var.pkl")
e_1 = load_pickle(run_path_1 / "E_var.pkl")

env_to_remove: list[HillsTerrain] = [
    HillsTerrain(2.0, 5),
    HillsTerrain(2.0, 10),
    HillsTerrain(2.0, 15),
    HillsTerrain(2.0, 20)
]

for i in reversed(range(len(e_1))):
    env = e_1[i][0]
    if env in env_to_remove:
        print(f"{i}: {env.__str__()}")
        del e_1[i]
        del g_1[i]

save_pickle("E_var.pkl", e_1, run_path_1)
save_pickle("G_var.pkl", g_1, run_path_1)

# g_2 = load_pickle(run_path_2 / "G_var.pkl")
# e_2 = load_pickle(run_path_2 / "E_var.pkl")

# g = g_1 + g_2
# print(len(g))
# save_pickle("G_var.pkl", g, run_path_1)

# e = e_1 + e_2
# save_pickle("E_var.pkl", e, run_path_1)
# print(len(e))