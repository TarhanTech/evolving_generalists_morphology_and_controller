"""This executable creates graphs, images and videos for the experimental runs"""

import os
from pathlib import Path
import argparse
import pickle
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from source.training_env import (
    TrainingSchedule,
    RoughTerrain,
    HillsTerrain,
    DefaultTerrain,
    TerrainType,
)
from source.individual import Individual
from source.globals import *


class GraphBuilder:
    """Class used to create graphs, images and videos for the experimental runs"""

    def __init__(self, run_paths: List[Path]):
        self.inds: List[Individual] = [Individual(id=i + 20) for i in range(6)]
        self.run_paths: List[Path] = run_paths
        
        self._path_numbers: int = len(self.run_paths)
        self._evaluation_count: int = 50

        self.g: List[List[torch.Tensor]] = self._load_g()
        self.e: List[List[List[TerrainType]]] = self._load_e()
        self._print_run_data()

        self.morph_data_dfs: list[list[pd.DataFrame]] = self._load_morph_data()
        self.env_fitnesses: List[List[Tuple[TerrainType, float]]] = self._evaluate_all_envs()


    def build_generalist_evaluation_graph(self):
        for i in range(len(self.run_paths)):
            gen_score_df = pd.read_csv(self.run_paths[i] / "gen_score_pandas_df.csv")
            gen_score_df["Generation"] = range(
                algo_init_training_generations,
                algo_init_training_generations + len(gen_score_df),
            )
            gen_score_df.set_index("Generation", inplace=True)
            plt.figure(figsize=(12, 6))

            plt.plot(
                gen_score_df.index,
                gen_score_df["Generalist Score"],
                label="Generalist Score",
                marker="o",
            )

            plt.xlabel("Generation")
            plt.ylabel("Generalist Scores")
            plt.title("Generalist Scores During Evolution")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(
                self.run_paths[i] / "generalist_score_metrics_plot.png",
                dpi=300,
                bbox_inches="tight",
            )

    def create_ant_screenshots(self):
        for i in range(self.path_numbers):
            for j in range(len(self.g[i])):
                self.inds[0].setup_ant_default(self.g[i][j])
                self.inds[0].make_screenshot_ant(self.run_paths[i] / f"partition_{j+1}" / "ant.png")

    def create_generalist_heatmap_partition(self):
        for i in range(self.path_numbers):
            g = self.g[i]
            e = self.e[i]
            rt_rows = np.round(
                np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step),
                1,
            )
            rt_columns = np.round(
                np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step),
                1,
            )
            rt_df = pd.DataFrame(index=rt_rows, columns=rt_columns, dtype=float)

            hills_rows = np.round(
                np.arange(
                    hills_scale_start,
                    hills_scale_end + hills_scale_step,
                    hills_scale_step,
                ),
                1,
            )
            hills_columns = np.round(
                np.arange(
                    hills_floor_start,
                    hills_floor_end + hills_floor_step,
                    hills_floor_step,
                ),
                1,
            )
            hills_df = pd.DataFrame(index=hills_rows, columns=hills_columns, dtype=float)

            default_df = pd.DataFrame(np.random.random(), index=[0], columns=[0])

            for j in range(len(e)):
                for env in e[j]:
                    if isinstance(env, RoughTerrain):
                        rt_df.loc[round(env.block_size, 1), round(env.floor_height, 1)] = j
                    elif isinstance(env, HillsTerrain):
                        hills_df.loc[round(env.scale, 1), round(env.floor_height, 1)] = j
                    elif isinstance(env, DefaultTerrain):
                        default_df.iloc[0, 0] = j
                    else:
                        assert False, "Class type not supported"

            plt.figure(figsize=(18, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])  # Adjust width ratios as needed

            vmin = 0
            vmax = len(g)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])

            # Hill Environment Heatmap
            sns.heatmap(
                hills_df,
                ax=ax0,
                annot=True,
                cbar=False,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax0.set_title("Hill Environment")
            ax0.set_xlabel("Floor Height")
            ax0.set_ylabel("Scale")

            # Rough Environment Heatmap
            sns.heatmap(
                rt_df,
                ax=ax1,
                annot=True,
                cbar=False,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax1.set_title("Rough Environment")
            ax1.set_xlabel("Floor Height")
            ax1.set_ylabel("Block Size")

            # Default Environment Heatmap
            heatmap = sns.heatmap(
                default_df,
                ax=ax2,
                annot=True,
                cbar=False,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            heatmap.set_xticklabels([])
            heatmap.set_yticklabels([])
            ax2.set_title("Default Environment")

            plt.tight_layout()  # Adjust layout
            plt.savefig(
                self.run_paths[i] / "generalist_heatmap_partition.png",
                dpi=300,
                bbox_inches="tight",
            )

    def create_fitness_heatmap(self):
        for i in range(self.path_numbers):
            env_fitnesses = self.env_fitnesses[i]
            rt_rows = np.round(
                np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step),
                1,
            )
            rt_columns = np.round(
                np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step),
                1,
            )
            rt_df = pd.DataFrame(index=rt_rows, columns=rt_columns, dtype=float)

            hills_rows = np.round(
                np.arange(
                    hills_scale_start,
                    hills_scale_end + hills_scale_step,
                    hills_scale_step,
                ),
                1,
            )
            hills_columns = np.round(
                np.arange(
                    hills_floor_start,
                    hills_floor_end + hills_floor_step,
                    hills_floor_step,
                ),
                1,
            )
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
            sns.heatmap(
                hills_df,
                ax=ax0,
                annot=True,
                cbar=False,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                fmt=".1f",
            )
            ax0.set_title("Hill Environment")
            ax0.set_xlabel("Floor Height")
            ax0.set_ylabel("Scale")
            for floor_height in tr_schedule.floor_heights_for_testing_hills:
                col_index = hills_df.columns.get_loc(floor_height)
                ax0.add_patch(
                    Rectangle(
                        (col_index, 0),
                        1,
                        len(hills_df),
                        fill=False,
                        edgecolor="red",
                        lw=5,
                    )
                )

            hills_mean_training = hills_df[[2.2, 2.4, 2.6, 3.4, 3.6, 3.8]].mean(axis=None)
            hills_std_training = pd.Series(hills_df[[2.2, 2.4, 2.6, 3.4, 3.6, 3.8]].values.flatten()).std()
            plt.figtext(
                0.25,
                0,
                f"Overall Mean Training: {hills_mean_training:.2f}, Overall STD Training: {hills_std_training:.2f}",
                ha="center",
            )
            hills_mean_testing = hills_df[tr_schedule.floor_heights_for_testing_hills].mean(axis=None)
            hills_std_testing = pd.Series(hills_df[tr_schedule.floor_heights_for_testing_hills].values.flatten()).std()
            plt.figtext(
                0.25,
                -0.05,
                f"Overall Mean Testing: {hills_mean_testing:.2f}, Overall STD Testing: {hills_std_testing:.2f}",
                ha="center",
            )
            plt.figtext(
                0.25,
                -0.1,
                f"Overall Mean: {hills_df.values.mean():.2f}, Overall STD: {hills_df.values.std():.2f}",
                ha="center",
            )

            # Rough Environment Heatmap
            sns.heatmap(
                rt_df,
                ax=ax1,
                annot=True,
                cbar=False,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                fmt=".1f",
            )
            ax1.set_title("Rough Environment")
            ax1.set_xlabel("Floor Height")
            ax1.set_ylabel("Block Size")
            for floor_height in tr_schedule.floor_heights_for_testing_rough:
                col_index = rt_df.columns.get_loc(floor_height)
                ax1.add_patch(Rectangle((col_index, 0), 1, len(rt_df), fill=False, edgecolor="red", lw=5))

            rt_mean_training = rt_df[[0.2, 0.3, 0.4, 0.7, 0.8, 0.9]].mean(axis=None)
            rt_std_training = pd.Series(rt_df[[0.2, 0.3, 0.4, 0.7, 0.8, 0.9]].values.flatten()).std()
            plt.figtext(
                0.7,
                0,
                f"Overall Mean Training: {rt_mean_training:.2f}, Overall STD Training: {rt_std_training:.2f}",
                ha="center",
            )
            rt_mean_testing = rt_df[tr_schedule.floor_heights_for_testing_rough].mean(axis=None)
            rt_std_testing = pd.Series(rt_df[tr_schedule.floor_heights_for_testing_rough].values.flatten()).std()
            plt.figtext(
                0.7,
                -0.05,
                f"Overall Mean Testing: {rt_mean_testing:.2f}, Overall STD Testing: {rt_std_testing:.2f}",
                ha="center",
            )
            plt.figtext(
                0.7,
                -0.1,
                f"Overall Mean: {rt_df.values.mean():.2f}, Overall STD: {rt_df.values.std():.2f}",
                ha="center",
            )

            # Default Environment Heatmap
            heatmap = sns.heatmap(
                default_df,
                ax=ax2,
                annot=True,
                cbar=True,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                fmt=".1f",
            )
            heatmap.set_xticklabels([])
            heatmap.set_yticklabels([])
            ax2.set_title("Default Environment")

            fitness_only = np.array([x[1] for x in env_fitnesses])
            mean_fitness = np.mean(fitness_only)
            std_fitness = np.std(fitness_only)
            plt.figtext(
                0.5,
                -0.2,
                f"Overall Mean: {mean_fitness:.2f}, Overall STD: {std_fitness:.2f}",
                ha="center",
            )

            plt.tight_layout()  # Adjust layout
            plt.savefig(self.run_paths[i] / "fitness_heatmap.png", dpi=300, bbox_inches="tight")

    def create_fitness_env_boxplot(self):
        for i in range(self.path_numbers):
            env_fitnesses = self.env_fitnesses[i]
            fitness_rough_terrain = [x[1] for x in env_fitnesses if isinstance(x[0], RoughTerrain)]
            fitness_hills_terrain = [x[1] for x in env_fitnesses if isinstance(x[0], HillsTerrain)]
            fitness_values = fitness_rough_terrain + fitness_hills_terrain
            labels = ["Rough Terrain"] * len(fitness_rough_terrain) + ["Hills Terrain"] * len(fitness_hills_terrain)

            sns.set(style="whitegrid")

            plt.figure(figsize=(10, 6))
            boxplot = sns.boxplot(
                x=labels,
                y=fitness_values,
                width=0.3,
                palette=["magenta", "teal"],
                hue=labels,
            )
            boxplot.set_title("Fitness Distribution by Environment", fontsize=16, fontweight="bold")
            boxplot.set_ylabel("Fitness", fontsize=14)
            boxplot.set_xlabel("Environment", fontsize=14)
            boxplot.tick_params(labelsize=12)

            plt.savefig(self.run_paths[i] / "fitness_boxplot.png", dpi=300, bbox_inches="tight")

    def create_boxplot_experiments(self):
        if self.path_numbers <= 1:
            return

        labels: List[str] = []
        fitness_values: List[float] = []
        for i in range(self.path_numbers):
            fitness_only: list[float] = [x[1] for x in self.env_fitnesses[i]]
            label: str = self.run_paths[i].name
            labels.extend([label] * len(fitness_only))
            fitness_values.extend(fitness_only)
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 6))
        boxplot = sns.boxplot(
            x=labels,
            y=fitness_values,
            width=0.45,
            palette=["magenta", "teal", "red"],
            hue=labels,
        )
        boxplot.set_title(
            "Fitness Distribution on all Environments by Experiment",
            fontsize=16,
            fontweight="bold",
        )
        boxplot.set_ylabel("Fitness", fontsize=14)
        boxplot.set_xlabel("Experiment", fontsize=14)
        boxplot.tick_params(labelsize=10)

        plt.savefig("./fitness_boxplot_experiments.png", dpi=300, bbox_inches="tight")

    def create_morph_params_plot(self):
        def _create_plot(df, generations, ylabel, save_path):
            plt.figure(figsize=(20, 4))
            for column in df.columns:
                plt.plot(generations, df[column], label=column)

            plt.title(f"Ant {ylabel} Morphology Changes Over Generations")
            plt.xlabel("Generation")
            plt.ylabel(ylabel)
            # Setting x-axis ticks to show every generation mark if needed or skip some for clarity
            if len(generations) > 10:
                tick_spacing = int(
                    len(generations) / 10
                )  # Shows a tick at every 10th generation if there are many points
                plt.xticks(generations[::tick_spacing])
            else:
                plt.xticks(generations)

            plt.xticks(rotation=45)  # Optional: rotate labels to improve readability
            plt.legend()
            plt.grid(True)  # Optional: adds grid lines for better readability
            plt.savefig(save_path)

        for i in range(self.path_numbers):
            for j, df in enumerate(self.morph_data_dfs[i]):
                folder_save_path: Path = self.run_paths[i] / f"partition_{j+1}/morph_params_evolution_plots"
                os.makedirs(folder_save_path, exist_ok=True)

                width_columns = [col for col in df.columns if "width" in col]
                length_columns = [col for col in df.columns if "length" in col]
                aux_width_columns = [col for col in width_columns if "aux" in col]
                ankle_width_columns = [col for col in width_columns if "ankle" in col]
                aux_length_columns = [col for col in length_columns if "aux" in col]
                ankle_length_columns = [col for col in length_columns if "ankle" in col]

                df_aux_width = df[aux_width_columns]
                df_ankle_width = df[ankle_width_columns]
                df_aux_length = df[aux_length_columns]
                df_ankle_length = df[ankle_length_columns]

                generations = np.arange(10, 10 * len(df) + 10, 10)

                _create_plot(
                    df_aux_width,
                    generations,
                    "Aux Leg Width",
                    folder_save_path / "aux_leg_width_plot.png",
                )
                _create_plot(
                    df_ankle_width,
                    generations,
                    "Ankle Leg Width",
                    folder_save_path / "ankle_leg_width_plot.png",
                )
                _create_plot(
                    df_aux_length,
                    generations,
                    "Aux Leg Length",
                    folder_save_path / "aux_leg_length_plot.png",
                )
                _create_plot(
                    df_ankle_length,
                    generations,
                    "Ankle Leg Length",
                    folder_save_path / "ankle_leg_length_plot.png",
                )

    def create_morph_params_pca_scatterplot(self):
        for i in range(self.path_numbers):
            for j, df in enumerate(self.morph_data_dfs[i]):
                folder_save_path: str = self.run_paths[i] / f"partition_{j+1}"
                scaler: StandardScaler = StandardScaler()
                df_scaled = scaler.fit_transform(df)

                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(df_scaled)

                df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
                df_pca["generations"] = list(range(10, len(df_pca) * 10 + 10, 10))

                plt.figure(figsize=(8, 6))

                scatter = plt.scatter(
                    df_pca["PC1"],
                    df_pca["PC2"],
                    c=df_pca["generations"],
                    cmap="viridis",
                )
                plt.colorbar(scatter, label="Generation Year")

                plt.xlabel("First Principal Component")
                plt.ylabel("Second Principal Component")
                plt.grid(True)
                plt.savefig(
                    folder_save_path / "pca_scatterplot.png",
                    dpi=300,
                    bbox_inches="tight",
                )

    def create_evolution_video(self):
        """Method creating evolution video by putting all images from screenshot folder back-to-back"""
        for i in range(self.path_numbers):
            for j in range(len(self.g)):
                partition_folder: Path = self.run_paths[i] / f"partition_{j+1}"
                images_folder: Path = partition_folder / "screenshots"

                images = [img for img in os.listdir(images_folder)]
                frame = cv2.imread(images_folder / images[0])
                height, width, layers = frame.shape

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(partition_folder / "evolution_video.mp4", fourcc, 30, (width, height))

                for image in images:
                    video.write(cv2.imread(os.path.join(images_folder, image)))

                cv2.destroyAllWindows()
                video.release()

    def _load_g(self):
        g = []
        for i in range(self.path_numbers):
            with open(self.run_paths[i] / "G_var.pkl", "rb") as file:
                g.append(pickle.load(file))
        return g

    def _load_e(self):
        e = []
        for i in range(self.path_numbers):
            with open(self.run_paths[i] / "E_var.pkl", "rb") as file:
                e.append(pickle.load(file))
        return e

    def _load_morph_data(self) -> list[list[pd.DataFrame]]:
        morph_data_dfs_runs: list[list[pd.DataFrame]] = []
        for i in range(self.path_numbers):
            morph_data_dfs: list[pd.DataFrame] = []

            for folder in os.listdir(self.run_paths[i]):
                full_path = self.run_paths[i] / folder

                if full_path.is_dir() and folder.startswith("partition"):
                    tensors_path: Path = full_path / "gen_tensors"
                    morph_data = []

                    for tensor_file in os.listdir(tensors_path):
                        tensor_path = tensors_path / tensor_file
                        params = torch.load(tensor_path)
                        self.inds[0].setup_ant_default(params)
                        morph_data.append(self.inds[0].mjEnv.morphology.morph_params_map)
                    morph_data_dfs.append(pd.DataFrame(morph_data))
            morph_data_dfs_runs.append(morph_data_dfs)
        return morph_data_dfs_runs

    def _print_run_data(self):
        for i in range(self.path_numbers):
            print(f"Data from run path: {self.run_paths[i]}")
            total_environments = sum(len(sublist) for sublist in self.e[i])
            print(f"Total generalist controllers: {len(self.g[i])}")
            print(f"Total environment partitions: {len(self.e[i])}")
            print(f"Total number of elements in E: {total_environments}\n")

    def _evaluate_all_envs(self) -> List[List[Tuple[TerrainType, float]]]:
        schedule = TrainingSchedule()
        env_fitnesses: List[List[Tuple[TerrainType, float]]] = []
        for i in range(self.path_numbers):
            g = self.g[i]
            e = self.e[i]
            for i in range(len(schedule.testing_schedule)):
                test_env = schedule.testing_schedule[i]
                index = self._decide_on_partition(e, test_env)
                if index != None:
                    e[index].append(test_env)

            env_fitness_results: List[Tuple[TerrainType, float]] = []
            for i in range(len(g)):
                params = g[i]
                env_partition = e[i]
                batch_size: int = len(self.inds)

                for j in range(0, len(env_partition), batch_size):
                    batch = env_partition[j : j + batch_size]
                    tasks = (joblib.delayed(self._evaluate)(env, ind, params) for env, ind in zip(batch, self.inds))
                    batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
                    env_fitness_results.extend(batch_fitness)
            env_fitnesses.append(env_fitness_results)
        return env_fitnesses

    def _decide_on_partition(self, e, test_env):
        for i in range(len(e)):
            partition = e[i]
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

    def _evaluate(self, training_env, ind: Individual, params: torch.Tensor):
        if isinstance(training_env, RoughTerrain):
            ind.setup_ant_rough(params, training_env.floor_height, training_env.block_size)
            video_save_path = (
                f"./videos_env/{type(training_env).__name__}_{training_env.block_size}_{training_env.floor_height}"
            )
        elif isinstance(training_env, HillsTerrain):
            ind.setup_ant_hills(params, training_env.floor_height, training_env.scale)
            video_save_path = (
                f"./videos_env/{type(training_env).__name__}_{training_env.scale}_{training_env.floor_height}"
            )
        elif isinstance(training_env, DefaultTerrain):
            ind.setup_ant_default(params)
            video_save_path = f"./videos_env/{type(training_env).__name__}"
        else:
            assert False, "Class type not supported"

        fitness_sum = 0
        for i in range(self._evaluation_count):
            if i == -1:
                fitness_sum = fitness_sum + ind.evaluate_fitness(
                    render_mode="rgb_array", video_save_path=video_save_path
                )
            else:
                fitness_sum = fitness_sum + ind.evaluate_fitness()

        fitness_mean = fitness_sum / self._evaluation_count
        return (training_env, fitness_mean)


def main():
    """
    Initial entry point of the executable. 
    Execute:
    python graph.py --run_paths path1 path2 path3
    """
    parser = argparse.ArgumentParser(
        description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch"
    )
    parser.add_argument(
        "--run_paths",
        nargs="+",
        type=Path,
        required=True,
        help="A list of paths to the run that you want to create combined graphs for.",
    )
    args = parser.parse_args()

    graph_builder: GraphBuilder = GraphBuilder(args.run_paths)
    graph_builder.build_generalist_evaluation_graph()
    graph_builder.create_ant_screenshots()
    graph_builder.create_generalist_heatmap_partition()
    graph_builder.create_fitness_heatmap()
    graph_builder.create_fitness_env_boxplot()
    graph_builder.create_boxplot_experiments()
    graph_builder.create_morph_params_plot()
    graph_builder.create_morph_params_pca_scatterplot()
    graph_builder.create_evolution_video()
    plt.close()


if __name__ == "__main__":
    main()
