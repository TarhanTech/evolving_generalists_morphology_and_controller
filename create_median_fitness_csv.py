import os
import pickle
import numpy as np
import csv
from collections import defaultdict

from source.training_env import DefaultTerrain, HillsTerrain, RoughTerrain, TerrainType

def load_env_fitnesses(folder):
    """Loads env_fitnesses_{test,training}.pkl if they exist in `folder`.
       Returns a dict: { environment_short_string -> list of fitness values }."""
    env_to_fitnesses = defaultdict(list)
    for filename in ("env_fitnesses_test.pkl", "env_fitnesses_training.pkl"):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = pickle.load(f)
            for terrain_obj, fitness in data:
                env_to_fitnesses[terrain_obj.short_string()].append(fitness)
    return env_to_fitnesses

def aggregate_runs(base_dir):
    """Iterates over all subfolders in base_dir and aggregates environment-fitness data."""
    aggregated = defaultdict(list)
    if not os.path.isdir(base_dir):
        print(f"[Warning] '{base_dir}' does not exist or is not a directory.")
        return aggregated
    run_folders = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    for subfolder in run_folders:
        folder_path = os.path.join(base_dir, subfolder)
        run_data = load_env_fitnesses(folder_path)
        for env_str, fits in run_data.items():
            aggregated[env_str].extend(fits)
    return aggregated

def parse_env_key(env_key):
    """
    Converts an environment short_string (e.g. "def", "ht(2.6,5)", "rt(0.3,1)")
    into (env_name, par1, par2).
    env_name: "Hill", "Rough", or "Default".
    par1: For Hill -> scale, for Rough -> block_size, for Default -> None.
    par2: For Hill/Rough -> floor_height, for Default -> None.
    """
    if env_key == "def":
        return ("Default", None, None)
    elif env_key.startswith("ht("):
        inside = env_key[3:-1]
        floor_s, scale_s = inside.split(",")
        floor_val = float(floor_s)
        scale_val = float(scale_s)
        return ("Hill", scale_val, floor_val)  # (name="Hill", par1=scale, par2=floor_height)
    elif env_key.startswith("rt("):
        inside = env_key[3:-1]
        floor_s, block_s = inside.split(",")
        floor_val = float(floor_s)
        block_val = float(block_s)
        return ("Rough", block_val, floor_val) # (name="Rough", par1=block_size, par2=floor_height)
    else:
        return ("Unknown", None, None)

def main():
    runs_info = [
        ("runs/FullGen-MorphEvo-Gen",         "FullGen-MorphEvo-Gen"),
        ("runs/OurAlgo-MorphEvo-Gen",         "OurAlgo-MorphEvo-Gen"),
        ("runs/OurAlgo-LargeMorph-Gen",       "OurAlgo-LargeMorph-Gen"),
        ("runs/Spec-MorphEvo-Long-old-backup","Spec-MorphEvo-Long-old-backup"),
    ]

    output_folder = "./_graphs_for_paper"
    os.makedirs(output_folder, exist_ok=True)

    for base_dir, run_name in runs_info:
        aggregated_data = aggregate_runs(base_dir)
        medians = {}
        for env_str, fits in aggregated_data.items():
            if len(fits) > 0:
                medians[env_str] = np.median(fits)

        csv_filename = f"{run_name}_median_each_env.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["env_name", "par1", "par2", "median_fitness"])
            for env_str, median_val in medians.items():
                name, p1, p2 = parse_env_key(env_str)
                writer.writerow([name, p1, p2, median_val])

        print(f"Created CSV: {csv_path}")

if __name__ == "__main__":
    main()
