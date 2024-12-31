from pathlib import Path
import torch
from torch import Tensor
from source.training_env import TrainingSchedule, TerrainType
import pickle

def vscode_sorting_key(path):
    # Natural sorting: split into parts (digits and text)
    import re
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]

def save_pickle(filename: str, data: any, save_path: Path):
    """Saves data to a pickle"""
    pickle_path: Path = save_path / filename
    with open(pickle_path, "wb") as file:
        pickle.dump(data, file)

g: list[Tensor] = []
e: list[list[TerrainType]] = []
t: TrainingSchedule = TrainingSchedule()

run_path: Path = Path("runs/Spec-MorphEvo-Long_19-12_21-57-24-152797")
specialist_path: Path = run_path / "specialist"

# Iterate over folders in VSCode sorting order
for env_folder in sorted(specialist_path.iterdir(), key=lambda p: (not p.is_dir(), vscode_sorting_key(p))):
    tensors_path: Path = env_folder / "gen_tensors"
    best_files = list(tensors_path.glob("*_best.pt"))
    if best_files:
        # Extract the highest-numbered tensor file
        best_tensor_path = max(best_files, key=lambda f: int(f.stem.split('_')[1]))
        best_tensor: Tensor = torch.load(best_tensor_path, weights_only=False)
        g.append(best_tensor)

        # Match folder name to TerrainType and check for exactly one match
        folder_name = env_folder.name
        matched_terrains = [terrain for terrain in t.all_terrains if str(terrain) == folder_name]
        if not matched_terrains:
            raise ValueError(f"No terrain matched for folder: {folder_name}")
        if len(matched_terrains) > 1:
            raise ValueError(f"Multiple terrains matched for folder: {folder_name}. Matches: {matched_terrains}")
        
        # Append the single matched environment
        e.append(matched_terrains)
print(e)
# Print lengths to verify
print(f"Number of tensors: {len(g)}")
print(f"Number of environments: {len(e)}")

# Save the tensors and environments
save_pickle("G_var.pkl", g, run_path)
save_pickle("E_var.pkl", e, run_path)
