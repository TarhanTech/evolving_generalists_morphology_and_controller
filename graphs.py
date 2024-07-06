import matplotlib.pyplot as plt
import pandas as pd
from source.training_env import *
import seaborn as sns
import argparse
import torch
from source.individual import *
from typing import List
import joblib
import pickle
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

def create_plot_gen_score(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df["Generalist Score"], label="Generalist Score", marker="o")

    plt.xlabel("Generation")
    plt.ylabel("Generalist Scores")
    plt.title("Generalist Scores During Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/generalist_score_metrics_plot.png", dpi=300, bbox_inches="tight")

def create_fitness_heatmap(env_fitnesses, save_path: str):
    rt_rows = np.round(np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step), 1)
    rt_columns = np.round(np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step), 1)
    rt_df = pd.DataFrame(index=rt_rows, columns=rt_columns, dtype=float)

    hills_rows = np.round(np.arange(hills_scale_start, hills_scale_end + hills_scale_step, hills_scale_step), 1)
    hills_columns = np.round(np.arange(hills_floor_start, hills_floor_end + hills_floor_step, hills_floor_step), 1)
    hills_df = pd.DataFrame(index=hills_rows, columns=hills_columns, dtype=float)

    default_df = pd.DataFrame(np.random.random(), index=[0], columns=[0])

    for env_fitness in env_fitnesses:
        env = env_fitness[0]
        fitness = env_fitness[1]
        if isinstance(env, RoughTerrain):
            rt_df.loc[round(env.block_size, 1), round(env.floor_height, 1)] = fitness
        elif isinstance(env, HillsTerrain):
            hills_df.loc[round(env.scale, 1), round(env.floor_height, 1)] = fitness
        elif isinstance(env, DefaultTerrain):
            default_df.iloc[0, 0] = fitness
        else:
            assert False, "Class type not supported"


    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])  # Adjust width ratios as needed

    vmin = 0
    vmax = 2000
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    # Hill Environment Heatmap
    sns.heatmap(hills_df, ax=ax0, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax0.set_title("Hill Environment")
    ax0.set_xlabel("Floor Height")
    ax0.set_ylabel("Scale")

    # Rough Environment Heatmap
    sns.heatmap(rt_df, ax=ax1, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax1.set_title("Rough Environment")
    ax1.set_xlabel("Floor Height")
    ax1.set_ylabel("Block Size")

    # Default Environment Heatmap
    heatmap = sns.heatmap(default_df, ax=ax2, annot=True, cbar=True, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    heatmap.set_xticklabels([])
    heatmap.set_yticklabels([])
    ax2.set_title("Default Environment")

    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{save_path}/fitness_heatmap.png", dpi=300, bbox_inches="tight")

def create_generalist_heatmap_partition(G, E, save_path: str):
    rt_rows = np.round(np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step), 1)
    rt_columns = np.round(np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step), 1)
    rt_df = pd.DataFrame(index=rt_rows, columns=rt_columns, dtype=float)

    hills_rows = np.round(np.arange(hills_scale_start, hills_scale_end + hills_scale_step, hills_scale_step), 1)
    hills_columns = np.round(np.arange(hills_floor_start, hills_floor_end + hills_floor_step, hills_floor_step), 1)
    hills_df = pd.DataFrame(index=hills_rows, columns=hills_columns, dtype=float)

    default_df = pd.DataFrame(np.random.random(), index=[0], columns=[0])

    for i in range(len(E)):
        for env in E[i]:
            if isinstance(env, RoughTerrain):
                rt_df.loc[round(env.block_size, 1), round(env.floor_height, 1)] = i
            elif isinstance(env, HillsTerrain):
                hills_df.loc[round(env.scale, 1), round(env.floor_height, 1)] = i
            elif isinstance(env, DefaultTerrain):
                default_df.iloc[0, 0] = i
            else:
                assert False, "Class type not supported"

    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])  # Adjust width ratios as needed

    vmin = 0
    vmax = len(G)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    # Hill Environment Heatmap
    sns.heatmap(hills_df, ax=ax0, annot=True, cbar=False, cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.set_title("Hill Environment")
    ax0.set_xlabel("Floor Height")
    ax0.set_ylabel("Scale")

    # Rough Environment Heatmap
    sns.heatmap(rt_df, ax=ax1, annot=True, cbar=False, cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("Rough Environment")
    ax1.set_xlabel("Floor Height")
    ax1.set_ylabel("Block Size")

    # Default Environment Heatmap
    heatmap = sns.heatmap(default_df, ax=ax2, annot=True, cbar=False, cmap="viridis", vmin=vmin, vmax=vmax)
    heatmap.set_xticklabels([])
    heatmap.set_yticklabels([])
    ax2.set_title("Default Environment")

    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{save_path}/generalist_heatmap_partition.png", dpi=300, bbox_inches="tight")

def evaluate(training_env, ind: Individual, params: torch.Tensor):
    if isinstance(training_env, RoughTerrain):
        ind.setup_ant_rough(params, training_env.floor_height, training_env.block_size)
    elif isinstance(training_env, HillsTerrain):
        ind.setup_ant_hills(params, training_env.floor_height, training_env.scale)
    elif isinstance(training_env, DefaultTerrain):
        ind.setup_ant_default(params)
    else:
        assert False, "Class type not supported"
    return (training_env, ind.evaluate_fitness())

def evaluate_training_env(individuals: List[Individual], params: torch.Tensor):
    batch_size: int = len(individuals)
    fitness_np = np.empty((0, 2), dtype=object)
    tr_schedule: TrainingSchedule = TrainingSchedule()
    for i in range(0, len(tr_schedule.training_schedule), batch_size):
        batch = tr_schedule.training_schedule[i:i + batch_size]
        tasks = (joblib.delayed(evaluate)(training_env, ind, params) for training_env, ind in zip(batch, individuals))
        batch_fitness = np.array(joblib.Parallel(n_jobs=batch_size)(tasks))
        fitness_np = np.vstack((fitness_np, batch_fitness))
    return fitness_np

def main():
    parser = argparse.ArgumentParser(description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch")
    parser.add_argument("--run_path", type=str, help="Path to the run that you want to create graphs for")
    args = parser.parse_args()

    assert args.run_path != None, "A --run_path must be specified."

    # params: torch.Tensor = torch.load(f"{args.run_path}/gen_tensors/generalist_best.pt")
    gen_evo_df = pd.read_csv(f"{args.run_path}/gen_score_pandas_df.csv")

    with open(f"{args.run_path}/G_var.pkl", "rb") as file:
        G = pickle.load(file)
    with open(f"{args.run_path}/E_var.pkl", "rb") as file:
        E = pickle.load(file)
        
    total_elements = sum(len(sublist) for sublist in E)
    print(f"Total generalist controllers: {len(G)}")
    print(f"Total number of elements in E: {total_elements}")  

    individuals: List[Individual] = [Individual(id=i+20) for i in range(6)]

    env_fitnesses = evaluate_training_env(individuals, G[0])
    create_fitness_heatmap(env_fitnesses, args.run_path)

    create_generalist_heatmap_partition(G, E, args.run_path)

    create_plot_gen_score(gen_evo_df, args.run_path)

    plt.close()

if __name__ == "__main__": main()