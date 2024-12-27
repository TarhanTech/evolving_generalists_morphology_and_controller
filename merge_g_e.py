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

run_path_1: Path = Path("runs/Spec-MorphEvo-Long_19-12_21-57-24-152797")
run_path_2: Path = Path("runs/Spec-MorphEvo-Long_25-12_14-55-47-844595")

g_1 = load_pickle(run_path_1 / "G_var.pkl")
e_1 = load_pickle(run_path_1 / "E_var.pkl")
g_2 = load_pickle(run_path_2 / "G_var.pkl")
e_2 = load_pickle(run_path_2 / "E_var.pkl")

g = g_1 + g_2
print(len(g))
save_pickle("G_var.pkl", g, run_path_1)

e = e_1 + e_2
save_pickle("E_var.pkl", e, run_path_1)
print(len(e))