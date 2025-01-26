import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import List, Tuple, Optional
from pathlib import Path
import multiprocessing

from source.training_env import (
    DefaultTerrain,
    HillsTerrain,
    RoughTerrain,
    TerrainType,
    TrainingSchedule,
)
from source.individual import Individual
from source.algo import Algo

###############################################################################
# Decide (dis_morph_evo, morph_type) based on the run_path string
###############################################################################
def decide_ind_params(run_path: Path) -> Tuple[bool, Optional[str]]:
    rp_str = str(run_path)
    if "FullGen-DefaultMorph-Gen" in rp_str:
        return (True, "default")
    elif "FullGen-MorphEvo-Gen" in rp_str:
        return (False, None)
    elif "OurAlgo-CustomMorph-Gen" in rp_str:
        return (True, "custom")
    elif "OurAlgo-DefaultMorph-Gen" in rp_str:
        return (True, "default")
    elif "OurAlgo-LargeMorph-Gen" in rp_str:
        return (True, "large")
    elif "OurAlgo-MorphEvo-Gen" in rp_str:
        return (False, None)
    elif "OurAlgo-MorphEvo-StartLarge-Gen" in rp_str:
        return (False, None)
    elif "Spec-DefaultMorph" in rp_str:
        return (True, "default")
    elif "Spec-MorphEvo" in rp_str:
        return (False, None)
    elif "Spec-MorphEvo-Long" in rp_str:
        return (False, None)
    return (False, None)

###############################################################################
# Evaluate method
###############################################################################
def evaluate(params: torch.Tensor,
             terrain: TerrainType,
             run_path: Path) -> Tuple[float, float]:
    dis_morph_evo, morph_type = decide_ind_params(run_path)
    parallel_jobs = 15
    fitnesses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inds = [
        Individual(
            device=device,
            morph_params_bounds_enc=Algo.morph_params_bounds_enc,
            penalty_growth_rate=Algo.penalty_growth_rate,
            penalty_scale_factor=Algo.penalty_scale_factor,
            penalty_scale_factor_err=Algo.penalty_scale_factor_err,
            dis_morph_evo=dis_morph_evo,
            morph_type=morph_type
        )
        for _ in range(parallel_jobs)
    ]
    for ind in inds:
        ind.setup_env_ind(params, terrain)

    def eval_fitness(ind: Individual) -> float:
        return ind.evaluate_fitness()

    for _ in range(0, 15, parallel_jobs):
        batch_size = min(parallel_jobs, 15 - len(fitnesses))
        tasks = (joblib.delayed(eval_fitness)(i) for i in inds[:batch_size])
        batch_fitness = joblib.Parallel(n_jobs=parallel_jobs)(tasks)
        fitnesses.extend(batch_fitness)

    fitnesses = np.array(fitnesses)
    return (fitnesses.mean(), fitnesses.std())


###############################################################################
# Specialist code
###############################################################################
def load_env_fitnesses_specialist(folder: str) -> dict:
    env_to_best = {}
    for fname in ["env_fitnesses_test.pkl", "env_fitnesses_training.pkl"]:
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        for entry in data:
            if len(entry) == 3:
                terrain_obj, fitness_val, idx = entry
            else:
                terrain_obj, fitness_val = entry
                idx = None
            short_str = terrain_obj.short_string()
            old_best, old_idx = env_to_best.get(short_str, (-np.inf, None))
            if fitness_val > old_best:
                env_to_best[short_str] = (fitness_val, idx)
    return env_to_best

def load_specialist_controller(folder: str):
    """
    If G_var.pkl was saved with 'pickle.dump(g_var, f)', we must use `pickle.load`.
    Then we move each tensor to CPU if needed.
    """
    e_p = os.path.join(folder, "E_var.pkl")
    g_p = os.path.join(folder, "G_var.pkl")
    if not (os.path.exists(e_p) and os.path.exists(g_p)):
        return None, None

    with open(e_p, "rb") as fe:
        e_var = pickle.load(fe)

    # load using standard Python pickle
    with open(g_p, "rb") as fg:
        g_var = pickle.load(fg)

    # Ensure each tensor is on CPU
    if isinstance(g_var, list):
        for i, tensor in enumerate(g_var):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                g_var[i] = tensor.cpu()

    return e_var, g_var

def find_best_specialist_for_env(env: TerrainType, spec_base_dir: str):
    best_fitness = -np.inf
    best_info = None
    env_str = env.short_string()
    run_folders = [
        d for d in os.listdir(spec_base_dir)
        if os.path.isdir(os.path.join(spec_base_dir, d))
    ]
    for subfolder in run_folders:
        folder_path = os.path.join(spec_base_dir, subfolder)
        best_dict = load_env_fitnesses_specialist(folder_path)
        if env_str in best_dict:
            fit_val, idx_val = best_dict[env_str]
            if fit_val > best_fitness:
                best_fitness = fit_val
                best_info = {
                    "folder": folder_path,
                    "fitness": fit_val,
                    "env_index": idx_val
                }
    return best_info

def get_specialist_controller(folder: str,
                              env_idx: Optional[int],
                              env_obj: TerrainType):
    e_var, g_var = load_specialist_controller(folder)
    if e_var is None or g_var is None:
        return None
    if env_idx is not None and env_idx < len(g_var):
        return g_var[env_idx]

    # fallback search by environment
    env_str = env_obj.short_string()
    for i, terrains in enumerate(e_var):
        if isinstance(terrains, list):
            if any(t.short_string() == env_str for t in terrains):
                if i < len(g_var):
                    return g_var[i]
        else:
            if terrains.short_string() == env_str:
                if i < len(g_var):
                    return g_var[i]
    return None

def run_specialist_evaluations(output_csv: str):
    SPEC_BASE_DIR = "./runs/Spec-MorphEvo-Long-old-backup"
    ts = TrainingSchedule()
    rows = []

    target_envs = [
        HillsTerrain(scale=20, floor_height=2.6),
        HillsTerrain(scale=5,  floor_height=2.6),
        HillsTerrain(scale=20, floor_height=3.6),
        HillsTerrain(scale=5,  floor_height=3.6),
        RoughTerrain(block_size=4, floor_height=0.3),
        RoughTerrain(block_size=1, floor_height=0.3),
        RoughTerrain(block_size=4, floor_height=0.8),
        RoughTerrain(block_size=1, floor_height=0.8),
        DefaultTerrain(),
    ]

    for env_obj in target_envs:
        best_info = find_best_specialist_for_env(env_obj, SPEC_BASE_DIR)
        if best_info is None:
            rows.append({
                "env": env_obj.short_string(),
                "env_par1": "",
                "env_par2": "",
                "controller": "NoneFound",
                "fitness": "None"
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            continue

        folder = best_info["folder"]
        idx_val = best_info["env_index"]
        best_controller = get_specialist_controller(folder, idx_val, env_obj)
        if best_controller is None:
            rows.append({
                "env": env_obj.short_string(),
                "env_par1": "",
                "env_par2": "",
                "controller": "NoneFound",
                "fitness": "None"
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            continue

        for terrain in ts.all_terrains:
            mean_f, std_f = evaluate(best_controller, terrain, Path(folder))
            if isinstance(terrain, HillsTerrain):
                p1 = terrain.scale
                p2 = terrain.floor_height
            elif isinstance(terrain, RoughTerrain):
                p1 = terrain.block_size
                p2 = terrain.floor_height
            else:
                p1, p2 = None, None

            rows.append({
                "env": terrain.short_string(),
                "env_par1": p1,
                "env_par2": p2,
                "controller": env_obj.short_string(),
                "fitness": mean_f
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[Specialist] Created CSV: {output_csv}")


###############################################################################
# Generalist: also loaded with pickle, then .cpu() as needed
###############################################################################
def load_ensemble_data(folder: str):
    e_path = os.path.join(folder, "E_established.pkl")
    g_path = os.path.join(folder, "G_var.pkl")
    if not (os.path.exists(e_path) and os.path.exists(g_path)):
        return None, None

    with open(e_path, "rb") as fe:
        E_established = pickle.load(fe)

    # G_var also saved with pickle.dump
    with open(g_path, "rb") as fg:
        G_var = pickle.load(fg)

    # Ensure CPU
    if isinstance(G_var, list):
        for i, tensor in enumerate(G_var):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                G_var[i] = tensor.cpu()

    return E_established, G_var

def find_best_generalist_for_env(env_obj: TerrainType, base_dir: str):
    best_fitness = -np.inf
    best_info = None
    env_str = env_obj.short_string()

    run_folders = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    for subfolder in run_folders:
        folder_path = os.path.join(base_dir, subfolder)
        E_est, G_var = load_ensemble_data(folder_path)
        if E_est is None or G_var is None:
            continue

        for i, env_list in enumerate(E_est):
            if not isinstance(env_list, list):
                continue
            if any(t.short_string() == env_str for t in env_list):
                mean_f, std_f = evaluate(G_var[i], env_obj, Path(folder_path))
                if mean_f > best_fitness:
                    best_fitness = mean_f
                    best_info = {
                        "folder": folder_path,
                        "fitness": best_fitness,
                        "env_index": i
                    }
    return best_info

def run_generalist_evaluations(base_dir: str, output_csv: str):
    ts = TrainingSchedule()
    rows = []

    target_envs = [
        HillsTerrain(scale=20, floor_height=2.6),
        HillsTerrain(scale=5,  floor_height=2.6),
        HillsTerrain(scale=20, floor_height=3.6),
        HillsTerrain(scale=5,  floor_height=3.6),
        RoughTerrain(block_size=4, floor_height=0.3),
        RoughTerrain(block_size=1, floor_height=0.3),
        RoughTerrain(block_size=4, floor_height=0.8),
        RoughTerrain(block_size=1, floor_height=0.8),
        DefaultTerrain(),
    ]

    for env_obj in target_envs:
        best_info = find_best_generalist_for_env(env_obj, base_dir)
        if best_info is None:
            rows.append({
                "env": env_obj.short_string(),
                "env_par1": "",
                "env_par2": "",
                "controller": "NoneFound",
                "fitness": "None"
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            continue

        folder = best_info["folder"]
        idx_val = best_info["env_index"]
        E_est, G_var = load_ensemble_data(folder)
        if E_est is None or G_var is None or idx_val >= len(G_var):
            rows.append({
                "env": env_obj.short_string(),
                "env_par1": "",
                "env_par2": "",
                "controller": "NoneFound",
                "fitness": "None"
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            continue

        best_controller = G_var[idx_val]

        for terrain in ts.all_terrains:
            mean_f, std_f = evaluate(best_controller, terrain, Path(folder))
            if isinstance(terrain, HillsTerrain):
                p1 = terrain.scale
                p2 = terrain.floor_height
            elif isinstance(terrain, RoughTerrain):
                p1 = terrain.block_size
                p2 = terrain.floor_height
            else:
                p1, p2 = None, None

            rows.append({
                "env": terrain.short_string(),
                "env_par1": p1,
                "env_par2": p2,
                "controller": env_obj.short_string(),
                "fitness": mean_f
            })
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[Generalist] Created CSV: {output_csv}")

###############################################################################
# Parallel tasks
###############################################################################
def specialist_task():
    out_csv = "./_graphs_for_paper/Spec-MorphEvo-Long-old-backup_on_all_environments.csv"
    run_specialist_evaluations(out_csv)

def gen_morph_task():
    base_dir = "./runs/OurAlgo-MorphEvo-Gen"
    out_csv = "./_graphs_for_paper/OurAlgo-MorphEvo-Gen_ensemble_on_all_environments.csv"
    run_generalist_evaluations(base_dir, out_csv)

def gen_large_task():
    base_dir = "./runs/OurAlgo-LargeMorph-Gen"
    out_csv = "./_graphs_for_paper/OurAlgo-LargeMorph-Gen_ensemble_on_all_environments.csv"
    run_generalist_evaluations(base_dir, out_csv)

if __name__ == "__main__":
    os.makedirs("./_graphs_for_paper", exist_ok=True)

    with multiprocessing.Pool(processes=3) as pool:
        results = [
            # pool.apply_async(specialist_task),
            pool.apply_async(gen_morph_task),
            # pool.apply_async(gen_large_task),
        ]
        for r in results:
            r.get()

    print("All parallel runs completed.")
