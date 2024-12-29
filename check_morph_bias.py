import pandas as pd
import torch
import pickle
from pathlib import Path
from torch import Tensor
from source.individual import Individual

def get_max_fitness_in_partition(partition_folder: Path) -> float:
    df = pd.read_csv(partition_folder / "gen_score_pandas_df.csv")
    return df["Generalist Score"].max()

def main():
    exp_path = Path('./runs/OurAlgo-LargeMorph-Gen')
    ind = Individual("cpu", (-0.1, 0.1), 1.03, 100, 1000, False, False)
    data_list = []
    for run_folder in sorted(exp_path.iterdir(), key=lambda p: p.name):
        if not run_folder.is_dir():
            continue
        row_data = {"Run Folder": run_folder.name}
        g_var_path = run_folder / "G_var.pkl"
        if not g_var_path.exists():
            raise FileNotFoundError(f"G_var.pkl not found in {run_folder}")
        with open(g_var_path, "rb") as f:
            g_var_data = pickle.load(f)
        partition_folders = sorted(
            [p for p in run_folder.iterdir() if p.is_dir() and p.name.startswith("partition_")],
            key=lambda p: p.name
        )
        for partition_dir in partition_folders:
            partition_num = partition_dir.name.split("_")[1]
            # tensor_path = partition_dir / "gen_tensors" / "tensor_1_best.pt"
            # if not tensor_path.exists():
            #     raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
            # params = torch.load(tensor_path)
            # ind.setup_ant_default(params)
            # morph_params_map = ind.mj_env.morphology.morph_params_map
            # for key_name, value in morph_params_map.items():
            #     row_data[f"P{partition_num} {key_name}"] = value
            row_data[f"P{partition_num} max fitness"] = get_max_fitness_in_partition(partition_dir)
            # final_params = g_var_data[int(partition_num) - 1]
            # ind.setup_ant_default(final_params)
            # final_morph_params_map = ind.mj_env.morphology.morph_params_map
            # for key_name, value in final_morph_params_map.items():
            #     row_data[f"End P{partition_num} {key_name}"] = value
        data_list.append(row_data)
    df = pd.DataFrame(data_list)
    print(df)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
