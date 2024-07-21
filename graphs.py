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
from matplotlib.patches import Rectangle
from source.globals import *

def create_plot_gen_score(df: pd.DataFrame, save_path: str):
    df["Generation"] = range(algo_init_training_generations, algo_init_training_generations + len(df))
    df.set_index("Generation", inplace=True)

    max_value = df.max()
    print(f"Max value: {max_value}")

    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df["Generalist Score"], label="Generalist Score", marker="o")

    plt.xlabel("Generation")
    plt.ylabel("Generalist Scores")
    plt.title("Generalist Scores During Evolution")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()

    plt.savefig(f"{save_path}/generalist_score_metrics_plot.png", dpi=300, bbox_inches="tight")

def create_fitness_boxplot(env_fitnesses, save_path: str):
    fitness_rough_terrain = [x[1] for x in env_fitnesses if isinstance(x[0], RoughTerrain)]
    fitness_hills_terrain = [x[1] for x in env_fitnesses if isinstance(x[0], HillsTerrain)]
    fitness_values = fitness_rough_terrain + fitness_hills_terrain
    labels = ["Rough Terrain"] * len(fitness_rough_terrain) + ["Hills Terrain"] * len(fitness_hills_terrain)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(x=labels, y=fitness_values, width=0.3, palette=["magenta", "teal"], hue=labels)
    boxplot.set_title("Fitness Distribution by Environment", fontsize=16, fontweight="bold")
    boxplot.set_ylabel("Fitness", fontsize=14)
    boxplot.set_xlabel("Environment", fontsize=14)
    boxplot.tick_params(labelsize=12)

    plt.savefig(f"{save_path}/fitness_boxplot.png", dpi=300, bbox_inches="tight")

def highlight_columns(df, columns_to_highlight):
    """ Create a mask for the columns to be highlighted """
    mask = np.full(df.shape, False)  # Initialize mask of False
    for col in columns_to_highlight:
        if col in df.columns:
            mask[:, df.columns.get_loc(col)] = True
    return mask

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
    vmax = 2500
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    tr_schedule = TrainingSchedule()
    # Hill Environment Heatmap
    sns.heatmap(hills_df, ax=ax0, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax0.set_title("Hill Environment")
    ax0.set_xlabel("Floor Height")
    ax0.set_ylabel("Scale")
    for floor_height in tr_schedule.floor_heights_for_testing_hills:
        col_index = hills_df.columns.get_loc(floor_height)
        ax0.add_patch(Rectangle((col_index, 0), 1, len(hills_df), fill=False, edgecolor='red', lw=5))

    # Rough Environment Heatmap
    sns.heatmap(rt_df, ax=ax1, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax1.set_title("Rough Environment")
    ax1.set_xlabel("Floor Height")
    ax1.set_ylabel("Block Size")
    for floor_height in tr_schedule.floor_heights_for_testing_rough:
        col_index = rt_df.columns.get_loc(floor_height)
        ax1.add_patch(Rectangle((col_index, 0), 1, len(rt_df), fill=False, edgecolor='red', lw=5))

    # Default Environment Heatmap
    heatmap = sns.heatmap(default_df, ax=ax2, annot=True, cbar=True, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    heatmap.set_xticklabels([])
    heatmap.set_yticklabels([])
    ax2.set_title("Default Environment")

    fitness_only = np.array([x[1] for x in env_fitnesses])
    mean_fitness = np.mean(fitness_only)
    std_fitness = np.std(fitness_only)

    plt.figtext(0.5, -0.02, f"Mean Fitness: {mean_fitness:.2f}, STD: {std_fitness:.2f}", ha="center")

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
    
    count = 50
    fitness_sum = 0
    for i in range(count):
        fitness_sum = fitness_sum + ind.evaluate_fitness()

    fitness_mean = fitness_sum / count
    return (training_env, fitness_mean)

def evaluate_G(individuals: List[Individual], params: torch.Tensor):
    fitness_np = np.empty((0, 2), dtype=object)
    batch_size: int = len(individuals)
    tr_schedule = TrainingSchedule()
    for j in range(0, len(tr_schedule.total_schedule), batch_size):
        batch = tr_schedule.total_schedule[j:j + batch_size]
        tasks = (joblib.delayed(evaluate)(env, ind, params) for env, ind in zip(batch, individuals))
        batch_fitness = np.array(joblib.Parallel(n_jobs=batch_size)(tasks))
        fitness_np = np.vstack((fitness_np, batch_fitness))
    return fitness_np

def decide_on_partition(E, test_env):
    for i in range(len(E)):
        partition = E[i]
        for env in partition:
            if type(env) == type(test_env):
                # look to the right cell, if there is nothing to the right, look at the left
                if isinstance(env, RoughTerrain):
                    if test_env.block_size == env.block_size:
                        if round((test_env.floor_height + rt_floor_step), 1) == env.floor_height:
                            return i
                        elif round((test_env.floor_height - rt_floor_step), 1) == env.floor_height:
                            return i
                elif isinstance(env, HillsTerrain):
                    if test_env.scale == env.scale:
                        if round((test_env.floor_height + hills_floor_step), 1) == env.floor_height:
                            return i
                        elif round((test_env.floor_height - hills_floor_step), 1) == env.floor_height:
                            return i
                else:
                    assert False, "Class type not supported"

def evaluate_training_env(individuals: List[Individual], G: List[torch.Tensor], E):
    schedule = TrainingSchedule()
    for i in range(len(schedule.testing_schedule)):
        test_env = schedule.testing_schedule[i]
        index = decide_on_partition(E, test_env)
        E[index].append(test_env)
    
    fitness_np = np.empty((0, 2), dtype=object)
    for i in range(len(G)):
        params = G[i]
        env_partition = E[i]
        batch_size: int = len(individuals)
        for j in range(0, len(env_partition), batch_size):
            batch = env_partition[j:j + batch_size]
            tasks = (joblib.delayed(evaluate)(env, ind, params) for env, ind in zip(batch, individuals))
            batch_fitness = np.array(joblib.Parallel(n_jobs=batch_size)(tasks))
            fitness_np = np.vstack((fitness_np, batch_fitness))
    return fitness_np

def main():
    parser = argparse.ArgumentParser(description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch")
    parser.add_argument("--run_path", type=str, help="Path to the run that you want to create graphs for")
    parser.add_argument("--tensor", type=str, help="Path to a tensor.pt file that should be tested")
    args = parser.parse_args()

    assert args.run_path != None or args.tensor != None, "A --run_path or --tensor must be specified."
    individuals: List[Individual] = [Individual(id=i+20) for i in range(6)]

    if args.tensor != None:
        folder_data_path = "./run_graphs"
        os.makedirs(folder_data_path, exist_ok=True)

        params = torch.load(args.tensor)

        individuals[0].setup_ant_default(params)
        individuals[0].make_screenshot_ant(f"{folder_data_path}/ant.png")
        
        env_fitnesses = evaluate_G(individuals, params)
        fitness_only = np.array([x[1] for x in env_fitnesses])
        print(f"Overall Mean: {np.mean(fitness_only)}")
        print(f"Overall STD: {np.std(fitness_only)}")

        create_fitness_boxplot(env_fitnesses, folder_data_path)
        create_fitness_heatmap(env_fitnesses, folder_data_path)
        plt.close()
    else:
        # params: torch.Tensor = torch.load(f"{args.run_path}/gen_tensors/generalist_best.pt")
        gen_evo_df = pd.read_csv(f"{args.run_path}/gen_score_pandas_df.csv")
        create_plot_gen_score(gen_evo_df, args.run_path)

        with open(f"{args.run_path}/G_var.pkl", "rb") as file:
            G = pickle.load(file)
        with open(f"{args.run_path}/E_var.pkl", "rb") as file:
            E = pickle.load(file)
            
        total_elements = sum(len(sublist) for sublist in E)
        print(f"Total generalist controllers: {len(G)}")
        print(f"Total number of elements in E: {total_elements}")  

        for i in range(len(G)): 
            individuals[0].setup_ant_default(G[i])
            individuals[0].make_screenshot_ant(f"{args.run_path}/ant_{i}.png")
        create_generalist_heatmap_partition(G, E, args.run_path)

        env_fitnesses = evaluate_training_env(individuals, G, E)
        fitness_only = np.array([x[1] for x in env_fitnesses])
        
        # env_fitnesses = evaluate_G(individuals, G[0])
        # fitness_only = np.array([x[1] for x in env_fitnesses])
        
        create_fitness_heatmap(env_fitnesses, args.run_path)
        create_fitness_boxplot(env_fitnesses, args.run_path)
        plt.close()

if __name__ == "__main__": main()