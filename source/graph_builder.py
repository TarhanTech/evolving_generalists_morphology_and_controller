"""This module contains graphbuilder classes"""
import copy
import os
import json
from pathlib import Path
import pickle
import re
from typing import List, Tuple
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
import torch
from torch import Tensor
import threading
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.stats import kruskal
import itertools

from source.training_env import (
    TrainingSchedule,
    RoughTerrain,
    HillsTerrain,
    DefaultTerrain,
    TerrainType,
)
from source.individual import Individual
from source.algo import Algo


class Graphbuilder(ABC):
    """Superclass used to create graphs for an experimental run, images and videos for the experimental runs"""

    def __init__(self, run_path: Path, dis_morph_evo: bool, morph_type: str):
        self.run_path: Path = run_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dis_morph_evo = dis_morph_evo
        self.inds: List[Individual] = [
            Individual(
                self.device,
                Algo.morph_params_bounds_enc,
                Algo.penalty_growth_rate,
                Algo.penalty_scale_factor,
                Algo.penalty_scale_factor_err,
                dis_morph_evo,
                morph_type
            )
            for _ in range(10)
        ]
        
        self.ts: TrainingSchedule = TrainingSchedule()

        self.g: List[Tensor] = self._load_g()
        self.e_init: List[List[TerrainType]] = self._load_e()
        self.e: List[List[TerrainType]] = [[] for _ in range(len(self.e_init))]

        self._print_run_data()

        self._evaluation_count: int = 30

    @abstractmethod
    def create_ant_screenshots(self):
        """Method that creates photos of the ants morphology in the environment"""
        pass

    def create_generalist_heatmap_partition(self, e: List[List[TerrainType]], file_name: str):
        """Method that creates heatmap to show which environments are being handled by which partition"""
        default_df, hills_df, rt_df = self._create_dataframe_terrains()

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
        vmax = len(self.g)
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
            self.run_path / file_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def create_fitness_heatmap(self):
        """Method that creates heatmap to show what the fitness score is per environment"""
        default_df, hills_df, rt_df = self._create_dataframe_terrains()

        for env_fitness in self.env_fitnesses:
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

        # Add red rectangles
        for test_terrain in self.ts.testing_terrains:
            if not isinstance(test_terrain, HillsTerrain):
                continue

            col_index = hills_df.columns.get_loc(test_terrain.floor_height)
            row_index = hills_df.index.get_loc(test_terrain.scale)
            ax0.add_patch(
                Rectangle(
                    (col_index, row_index),
                    1,
                    len(hills_df),
                    fill=False,
                    edgecolor="red",
                    lw=5,
                )
            )

        hills_training_fitnesses = [fitness for terrain, fitness in self.env_fitnesses_training if isinstance(terrain, HillsTerrain)]
        hills_mean_training = np.mean(hills_training_fitnesses)
        hills_std_training = np.std(hills_training_fitnesses)
        plt.figtext(
            0.25,
            0,
            f"Overall Mean Training: {hills_mean_training:.2f}, Overall STD Training: {hills_std_training:.2f}",
            ha="center",
        )
        hills_testing_fitnesses = [fitness for terrain, fitness in self.env_fitnesses_test if isinstance(terrain, HillsTerrain)]
        hills_mean_testing = np.mean(hills_testing_fitnesses)
        hills_std_testing = np.std(hills_testing_fitnesses)
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

        # Add red rectangles
        for test_terrain in self.ts.testing_terrains:
            if not isinstance(test_terrain, RoughTerrain):
                continue

            col_index = rt_df.columns.get_loc(test_terrain.floor_height)
            row_index = rt_df.index.get_loc(test_terrain.block_size)
            ax1.add_patch(
                Rectangle(
                    (col_index, row_index),
                    1,
                    len(rt_df),
                    fill=False,
                    edgecolor="red",
                    lw=5,
                )
            )

        rt_training_fitnesses = [fitness for terrain, fitness in self.env_fitnesses_training if isinstance(terrain, RoughTerrain)]
        rt_mean_training = np.mean(rt_training_fitnesses)
        rt_std_training = np.std(rt_training_fitnesses)
        plt.figtext(
            0.7,
            0,
            f"Overall Mean Training: {rt_mean_training:.2f}, Overall STD Training: {rt_std_training:.2f}",
            ha="center",
        )
        rt_testing_fitnesses = [fitness for terrain, fitness in self.env_fitnesses_test if isinstance(terrain, RoughTerrain)]
        rt_mean_testing = np.mean(rt_testing_fitnesses)
        rt_std_testing = np.std(rt_testing_fitnesses)
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

        fitness_only = np.array([x[1] for x in self.env_fitnesses])
        mean_fitness = np.mean(fitness_only)
        std_fitness = np.std(fitness_only)
        plt.figtext(
            0.5,
            -0.2,
            f"Overall Mean: {mean_fitness:.2f}, Overall STD: {std_fitness:.2f}",
            ha="center",
        )

        plt.tight_layout()  # Adjust layout
        plt.savefig(self.run_path / "fitness_heatmap.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def create_fitness_env_boxplot(self):
        """Method that creates boxplot of the fitness for each terrain type in one graph"""
        fitness_rough_terrain = [x[1] for x in self.env_fitnesses if isinstance(x[0], RoughTerrain)]
        fitness_hills_terrain = [x[1] for x in self.env_fitnesses if isinstance(x[0], HillsTerrain)]
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

        plt.savefig(self.run_path / "fitness_boxplot.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _load_g(self):
        with open(self.run_path / "G_var.pkl", "rb") as file:
            g = pickle.load(file)
        return g

    def _load_e(self) -> List[List[TerrainType]]:
        with open(self.run_path / "E_var.pkl", "rb") as file:
            e = pickle.load(file)
        return e

    def _print_run_data(self):
        print(f"Data from run path: {self.run_path}")
        total_environments = sum(len(sublist) for sublist in self.e)
        print(f"Total generalist controllers: {len(self.g)}")
        print(f"Total environment partitions: {len(self.e)}")
        print(f"Total number of elements in E: {total_environments}\n")

    def _create_dataframe_terrains(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rt_rows = np.round(
            np.arange(
                self.ts.rt_block_range[0],
                self.ts.rt_block_range[1] + self.ts.rt_block_step,
                self.ts.rt_block_step,
            ),
            1,
        )
        rt_columns = np.round(
            np.arange(
                self.ts.rt_floor_range[0],
                self.ts.rt_floor_range[1] + self.ts.rt_floor_step,
                self.ts.rt_floor_step,
            ),
            1,
        )
        rt_df = pd.DataFrame(index=rt_rows, columns=rt_columns, dtype=float)

        hills_rows = np.round(
            np.arange(
                self.ts.hills_scale_range[0],
                self.ts.hills_scale_range[1] + self.ts.hills_scale_step,
                self.ts.hills_scale_step,
            ),
            1,
        )
        hills_columns = np.round(
            np.arange(
                self.ts.hills_floor_range[0],
                self.ts.hills_floor_range[1] + self.ts.hills_floor_step,
                self.ts.hills_floor_step,
            ),
            1,
        )
        hills_df = pd.DataFrame(index=hills_rows, columns=hills_columns, dtype=float)

        default_df = pd.DataFrame(np.random.random(), index=[0], columns=[0])

        return default_df, hills_df, rt_df

    def _evaluate_envs(self, terrains: List[TerrainType], create_videos: bool) -> List[List[Tuple[TerrainType, float]]]:
        env_fitnesses: List[Tuple[TerrainType, float]] = []

        for terrain in terrains:
            env_fitness = self._evaluate(terrain, create_videos)
            env_fitnesses.append(env_fitness)
        return env_fitnesses

    def _evaluate(self, terrain: TerrainType,  create_videos: bool):
        def eval(ind: Individual):
            return ind.evaluate_fitness()

        fitnesses_part: list[float] = []
        for params in self.g:
            for ind in self.inds: ind.setup_env_ind(params, terrain) 
            fitnesses: list[float] = []
            batch_size = len(self.inds)
            for i in range(0, self._evaluation_count, batch_size):
                tasks = (joblib.delayed(eval)(ind) for ind in self.inds)
                batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
                fitnesses.extend(batch_fitness)

            mean_fitness = sum(fitnesses) / len(fitnesses)
            fitnesses_part.append(mean_fitness)

        highest_fitness = max(fitnesses_part)
        highest_fitness_index = fitnesses_part.index(highest_fitness)

        self.e[highest_fitness_index].append(copy.copy(terrain))

        if create_videos:
            video_thread = threading.Thread(
                target=self._create_video,
                args=(terrain, copy.deepcopy(self.inds[0]), self.g[highest_fitness_index])
            )
            video_thread.start()

        return (terrain, highest_fitness)

    def _create_video(self, terrain, ind: Individual, params):
        ind.setup_env_ind(params, terrain)

        video_save_path = self.run_path / "videos_env" / terrain.__str__()
        ind.evaluate_fitness(render_mode="rgb_array", video_save_path=video_save_path)

    def _change_folder_name(self):
        # Calculate mean fitness
        fitness_only = np.array([x[1] for x in self.env_fitnesses])
        mean_fitness = round(np.mean(fitness_only))

        # Current run path and folder name
        current_run_path = self.run_path
        folder_name = current_run_path.name

        # Define the regex pattern to capture an existing mean fitness number (if present)
        # and the timestamp
        pattern = r"^(.*?_)(\d+_)?(\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+)$"

        # Replacement function to insert or replace the mean fitness
        def repl(match):
            prefix = match.group(1)  # Everything before the existing mean fitness
            timestamp = match.group(3)  # The timestamp part
            return f"{prefix}{mean_fitness}_{timestamp}"

        # Apply the substitution to clean up and insert the new mean fitness
        new_folder_name = re.sub(pattern, repl, folder_name, count=1)

        # Construct the new run path
        new_run_path = current_run_path.parent / new_folder_name

        # Rename the folder if the new path doesn't already exist
        if not new_run_path.exists():
            current_run_path.rename(new_run_path)
            self.run_path = new_run_path
            print(f"Folder renamed to: {new_folder_name}")
        else:
            print(f"Cannot rename: {new_run_path} already exists.")


class GraphBuilderGeneralist(Graphbuilder):
    """Class used to create graphs, images and videos for the experimental runs dedicated for generalist runs"""

    def __init__(self, run_path: Path, create_videos: bool = False, dis_morph_evo = False, morph_type: str = None):
        super().__init__(run_path, dis_morph_evo, morph_type)

        self.e_init: List[List[TerrainType]] = self._load_e()
        self.e: List[List[TerrainType]] = [[] for _ in range(len(self.e_init))]

        test_file_path = self.run_path / "env_fitnesses_test.pkl"
        training_file_path = self.run_path / "env_fitnesses_training.pkl"
        e_file_path = self.run_path / "E_established.pkl"

        if os.path.exists(test_file_path) and os.path.exists(training_file_path) and os.path.exists(e_file_path):
            with open(test_file_path, "rb") as file:
                self.env_fitnesses_test = pickle.load(file)

            with open(training_file_path, "rb") as file:
                self.env_fitnesses_training = pickle.load(file)

            with open(e_file_path, "rb") as file:
                self.e = pickle.load(file)
            
            if create_videos is True:
                for i in range(len(self.e)):
                    for j in range(len(self.e[i])):
                        self._create_video(self.e[i][j], self.inds[0], self.g[i])

        else:
            self.env_fitnesses_test: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.testing_terrains, create_videos)
            with open(test_file_path, "wb") as file:
                pickle.dump(self.env_fitnesses_test, file)
            
            self.env_fitnesses_training: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.training_terrains, create_videos)
            with open(training_file_path, "wb") as file:
                pickle.dump(self.env_fitnesses_training, file)

            with open(e_file_path, "wb") as file:
                pickle.dump(self.e, file)


        self.env_fitnesses: List[Tuple[TerrainType, float]] = self.env_fitnesses_test + self.env_fitnesses_training
        
        self._change_folder_name()
        self._print_run_data()
        
        self.morph_step_size = 10 
        
        if dis_morph_evo is False:
            x1, x2, x3 = self._load_morph_data()
            self.morph_data_dfs: list[pd.DataFrame] = x1
            self.best_tensors_indices: list[list[int]] = x2
            self.best_images = x3
    
    def create_graphs(self):
        self.create_ant_screenshots()
        self.create_generalist_heatmap_partition(self.e_init, "hm_partition_by_algo.pdf")
        self.create_generalist_heatmap_partition(self.e, "hm_partition_by_best_mc.pdf")
        self.create_fitness_heatmap()
        self.create_fitness_env_boxplot()
        self.create_generalist_evaluation_graph()
        self.create_fitness_evaluation_graphs()
        self.create_morph_params_plot()
        self.create_morph_params_pca_scatterplot()
        self.create_evolution_video()

    def create_ant_screenshots(self):
        for i, g in enumerate(self.g):
            self.inds[0].setup_ant_default(g)
            self.inds[0].make_screenshot_ant(self.run_path / f"partition_{i+1}" / f"ant_{i+1}.png")
        print("Created Ant Screenshots")

    def create_generalist_evaluation_graph(self):
        """Method that creates a graph showing the generalist score of the best MC-pair in the generation"""
        for i, _ in enumerate(self.g):
            gen_score_df = pd.read_csv(self.run_path / f"partition_{i+1}" / "gen_score_pandas_df.csv")
            gen_score_df["Generation"] = range(len(gen_score_df))
            gen_score_df.set_index("Generation", inplace=True)
            plt.figure(figsize=(12, 6))

            # Plot the generalist score line
            plt.plot(
                gen_score_df.index,
                gen_score_df["Generalist Score"],
                label="Generalist Score",
                marker="o",
            )

            # Calculate and plot the maximum increasing line
            max_values = gen_score_df["Generalist Score"].cummax()
            plt.plot(
                gen_score_df.index,
                max_values,
                label="Maximum Increasing Line",
                linestyle="-",
                color="red",
            )

            plt.xlabel("Generation")
            plt.ylabel("Generalist Scores")
            plt.title("Generalist Scores During Evolution")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(
                self.run_path / f"partition_{i+1}" / "generalist_score_metrics_plot.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def create_fitness_evaluation_graphs(self):
        for i, _ in enumerate(self.g):
            json_file_path = self.run_path / f"partition_{i+1}" / "fitness_scores.json"
            with open(json_file_path, "r") as file:
                data = json.load(file)

            num_generations: int = 0
            for _, values in data.items():
                num_generations += len(values)


            fitness_evals_envs_path = self.run_path / f"partition_{i+1}" / "fitness_evals_envs"
            fitness_evals_envs_path.mkdir(parents=True, exist_ok=True)
            for key, values in data.items():
                original_indices = np.linspace(0, num_generations, num=len(values))

                interpolator = interp1d(original_indices, values, kind="linear")
                new_indices = np.linspace(0, num_generations, num=num_generations)
                interpolated_values = interpolator(new_indices)
                plt.figure()
                plt.plot(new_indices, interpolated_values)
                plt.title(f"{key} Plot")
                plt.xlabel("Generations")
                plt.ylabel("Fitness")
                plt.grid(True)
                plt.savefig(
                    fitness_evals_envs_path / f"{str(key)}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def create_morph_params_plot(self):
        """Method that creates graphs showing the change of morphological parameters over the generations"""
        if self.dis_morph_evo: return

        def _create_plot(df: pd.DataFrame, generations, ylabel, save_path):
            plt.figure(figsize=(20, 4))
            for column in df.columns:
                plt.plot(generations, df[column], label=column)

            plt.title(f"Ant {ylabel} Morphology Changes Over Generations")
            plt.xlabel("Generation")
            plt.ylabel(ylabel)

            if len(generations) > 10:
                tick_spacing = int(len(generations) / 10)
                plt.xticks(generations[::tick_spacing])
            else:
                plt.xticks(generations)

            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.savefig(save_path)

        for i, df in enumerate(self.morph_data_dfs):
            folder_save_path: Path = self.run_path / f"partition_{i+1}/morph_params_evolution_plots"
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

            generations = np.arange(1, len(df) * self.morph_step_size + 1, self.morph_step_size)
            _create_plot(
                df_aux_width,
                generations,
                "Aux Leg Width",
                folder_save_path / "aux_leg_width_plot.pdf",
            )
            _create_plot(
                df_ankle_width,
                generations,
                "Ankle Leg Width",
                folder_save_path / "ankle_leg_width_plot.pdf",
            )
            _create_plot(
                df_aux_length,
                generations,
                "Aux Leg Length",
                folder_save_path / "aux_leg_length_plot.pdf",
            )
            _create_plot(
                df_ankle_length,
                generations,
                "Ankle Leg Length",
                folder_save_path / "ankle_leg_length_plot.pdf",
            )
        plt.close()

    def create_morph_params_pca_scatterplot(self):
        """Method that creates a scatterplot of the morphological parameters which are reduced using PCA, showing the change in morphology over generations"""
        if self.dis_morph_evo: return

        def create_scatter_plot(x, y, c, x_label, y_label, c_label, save_path, best_x=None, best_y=None, images=None):
            # Create a figure with extended width to accommodate the images
            fig, ax = plt.subplots(figsize=(10, 6))  # Increased width to make room for images on the right

            # Create scatter plot
            scatter = ax.scatter(x=x, y=y, c=c, cmap="viridis")
            plt.colorbar(scatter, label=c_label)

            # Plot the line if best_x and best_y are provided
            if best_x is not None and best_y is not None:
                ax.plot(best_x, best_y, color="red", linewidth=2, label="Best Tensors")
                ax.legend()

                # Extend the x-axis limits to make room for the images outside the plot
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.6)  # Extend the x-axis to the right

                # Add 5 evenly spread images from the list if provided
                if images is not None:
                    num_images = min(len(best_x), len(images))  # Get the minimum between points and images
                    indices = np.round(np.linspace(0, num_images - 1, 5)).astype(int)  # 5 evenly spaced indices

                    # Calculate vertical positions for the images (evenly spaced)
                    image_y_positions = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5)

                    for idx, i in enumerate(indices):
                        x_coord = best_x.iloc[i] if hasattr(best_x, "iloc") else best_x[i]
                        y_coord = best_y.iloc[i] if hasattr(best_y, "iloc") else best_y[i]

                        # Position the image to the right, at a fixed horizontal location
                        x_offset = xlim[1] + (xlim[1] - xlim[0]) * 0.5  # Fixed offset to the right of the plot
                        y_offset = image_y_positions[idx]  # Vertically aligned based on index

                        # Place the image
                        img = OffsetImage(images[i], zoom=0.2)
                        ab = AnnotationBbox(img, (x_offset, y_offset), frameon=False, clip_on=False)
                        ax.add_artist(ab)

                        # Draw the arrow from the scatter point to the image
                        ax.annotate(
                            "",
                            xy=(x_coord, y_coord),
                            xytext=(x_offset, y_offset),
                            arrowprops=dict(color="black", arrowstyle="->")
                        )

            # Set labels and grid
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            # Close the plot to free memory
            plt.close()

        for i, df in enumerate(self.morph_data_dfs):
            gen_score_df = pd.read_csv(self.run_path / f"partition_{i+1}" / "gen_score_pandas_df.csv")
            folder_save_path: str = self.run_path / f"partition_{i+1}" / "pca_plots"
            os.makedirs(folder_save_path, exist_ok=True)

            scaler: StandardScaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            pca = PCA(n_components=1)
            pca_components = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_components, columns=["PC1"])
            df_pca["Generation"] = df.index

            gen_scores = []
            for j in df.index.to_list():
                gen_scores.append(gen_score_df.loc[j - 1, "Generalist Score"])
            df_pca["Generalist Score"] = gen_scores

            best_indices = self.best_tensors_indices[i] 
            best_tensors = df_pca.iloc[best_indices]

            create_scatter_plot(
                df_pca["Generalist Score"],
                df_pca["PC1"],
                df_pca["Generation"],
                "Generalist Score",
                "Principal Component Morphology",
                "Generations",
                folder_save_path / "one_pca_scatterplot.pdf",
                best_x=best_tensors["Generalist Score"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )

            # PCA with 2 components for the second scatterplot
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
            df_pca["Generation"] = df.index

            gen_scores = []
            for j in df.index.to_list():
                gen_scores.append(gen_score_df.loc[j - 1, "Generalist Score"])
            df_pca["Generalist Score"] = gen_scores

            best_tensors = df_pca.iloc[best_indices]

            create_scatter_plot(
                df_pca["PC2"],
                df_pca["PC1"],
                df_pca["Generation"],
                "2nd Principal Component Morphology",
                "1st Principal Component Morphology",
                "Generations",
                folder_save_path / "two_pca_generation_scatterplot.pdf",
                best_x=best_tensors["PC2"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )
            create_scatter_plot(
                df_pca["PC2"],
                df_pca["PC1"],
                df_pca["Generalist Score"],
                "2nd Principal Component Morphology",
                "1st Principal Component Morphology",
                "Generalist Score",
                folder_save_path / "two_pca_generalist_score_scatterplot.pdf",
                best_x=best_tensors["PC2"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )

    def create_evolution_video(self):
        """Method creating evolution video by putting all images from screenshot folder back-to-back"""
        for i in range(len(self.g)):
            partition_folder: Path = self.run_path / f"partition_{i+1}"
            images_folder: Path = partition_folder / "screenshots"
            sorted_image_files = sorted(os.listdir(images_folder), key=lambda file: get_creation_time(file, images_folder))

            frame = cv2.imread(str(images_folder / sorted_image_files[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(partition_folder / "evolution_video.mp4", fourcc, 30, (width, height))

            for image in sorted_image_files:
                video.write(cv2.imread(os.path.join(images_folder, image)))

            cv2.destroyAllWindows()
            video.release()
        
    def _load_morph_data(self) -> Tuple[list[pd.DataFrame], list[list[int]], list]:
        morph_data_dfs: list[pd.DataFrame] = []
        best_tensors_indices: list[list[int]] = []
        best_images_part = []

        for i, _ in enumerate(self.g):
            tensors_path = self.run_path / f"partition_{i+1}" / "gen_tensors"
            morph_data = []
            best_tensors_index = []
            best_images = []

            sorted_tensor_files = sorted(os.listdir(tensors_path), key=lambda file: get_creation_time(file, tensors_path))

            for j, tensor_file in enumerate(sorted_tensor_files):
                if j % self.morph_step_size == 0 or tensor_file.endswith("best.pt"):
                    tensor_path = tensors_path / tensor_file
                    params = torch.load(tensor_path, weights_only=False, map_location=torch.device('cpu'))
                    self.inds[0].setup_ant_default(params)
                    
                    morph_data.append({
                        **self.inds[0].mj_env.morphology.morph_params_map,
                        "Generation": j + 1
                        })
                    if tensor_file.endswith("best.pt"):
                        best_tensors_index.append(len(morph_data) - 1)
                        best_images.append(Image.open(self.run_path / f"partition_{i+1}" / "screenshots" / f"ant_{j+1}.png"))
            
            morph_data = pd.DataFrame(morph_data).set_index("Generation")
            morph_data_dfs.append(morph_data)
            best_tensors_indices.append(best_tensors_index)
            best_images_part.append(best_images)
        return (morph_data_dfs, best_tensors_indices, best_images_part)


class GraphBuilderSpecialist(Graphbuilder):
    """Class used to create graphs, images and videos for the experimental runs dedicated for specialist runs"""

    def __init__(self, run_path: Path, create_videos: bool = False, dis_morph_evo = False, morph_type: str = None):
        super().__init__(run_path, dis_morph_evo, morph_type)

        self.e: List[List[TerrainType]] = self._load_e()

        self.ts.training_terrains = self.ts.all_terrains
        self.ts.testing_terrains = []

        self.env_fitnesses_test: List[Tuple[TerrainType, float]] = []

        training_file_path = self.run_path / "env_fitnesses_training.pkl"
        if os.path.exists(training_file_path):
            with open(training_file_path, "rb") as file:
                self.env_fitnesses_training = pickle.load(file)
            print("Evaluations of env_fitnesses found in folder, reusing same one.")

            if create_videos is True:
                for i in range(len(self.e)):
                    for j in range(len(self.e[i])):
                        self._create_video(self.e[i][j], self.inds[0], self.g[i])
        else:
            self.env_fitnesses_training: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.training_terrains, create_videos)
            with open(training_file_path, "wb") as file:
                pickle.dump(self.env_fitnesses_training, file)
                print("evaluating env_fitnesses again")
        
        self.env_fitnesses: List[Tuple[TerrainType, float]] = self.env_fitnesses_test + self.env_fitnesses_training

        self._change_folder_name()
        self._print_run_data()

        self.morph_step_size = 10 

        x1, x2, x3 = self._load_morph_data()
        self.morph_data_dfs: list[pd.DataFrame] = x1
        self.best_tensors_indices: list[list[int]] = x2
        self.best_images = x3

    def create_graphs(self):
        self.create_ant_screenshots()
        self.create_generalist_heatmap_partition(self.e, "hm_partition.pdf")
        self.create_fitness_heatmap()
        self.create_fitness_env_boxplot()
        self.create_fitness_evaluation_graph()
        self.create_morph_params_plot()
        self.create_morph_params_pca_scatterplot()
        self.create_evolution_video()

    def create_ant_screenshots(self):
        for g, e in zip(self.g, self.e):
            self.inds[0].setup_ant_default(g)
            self.inds[0].make_screenshot_ant(self.run_path / "specialist" / e[0].__str__() / f"ant.png")
        print("Created Ant Screenshots")
        
    def create_fitness_evaluation_graph(self):
        for folder in os.listdir(self.run_path / "specialist"):
            full_folder_path: Path = self.run_path / "specialist" / folder
            df = pd.read_csv(full_folder_path / "pandas_logger_df.csv")
            df["Generation"] = range(1, len(df) + 1)
            df.set_index("Generation", inplace=True)

            plt.figure(figsize=(12, 6))

            plt.plot(
                df.index,
                df["pop_best_eval"],
                label="Fitness Score",
                marker="o",
            )

            plt.xlabel("Generation")
            plt.ylabel("Fitness Scores")
            plt.title("Fitness Scores During Evolution")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(
                full_folder_path / "fitness_score_metrics_plot.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

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
                tick_spacing = int(len(generations) / 10)  # Shows a tick at every 10th generation if there are many points
                plt.xticks(generations[::tick_spacing])
            else:
                plt.xticks(generations)

            plt.xticks(rotation=45)  # Optional: rotate labels to improve readability
            plt.legend()
            plt.grid(True)  # Optional: adds grid lines for better readability
            plt.savefig(save_path)
            plt.close()

        spec_folders = sorted(os.listdir(self.run_path / "specialist"), key=lambda file: get_creation_time(file, self.run_path / "specialist"))

        for df, folder in zip(self.morph_data_dfs, spec_folders):
            folder_save_path: Path = self.run_path / "specialist" / folder / "morph_params_evolution_plots"
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
                folder_save_path / "aux_leg_width_plot.pdf",
            )
            _create_plot(
                df_ankle_width,
                generations,
                "Ankle Leg Width",
                folder_save_path / "ankle_leg_width_plot.pdf",
            )
            _create_plot(
                df_aux_length,
                generations,
                "Aux Leg Length",
                folder_save_path / "aux_leg_length_plot.pdf",
            )
            _create_plot(
                df_ankle_length,
                generations,
                "Ankle Leg Length",
                folder_save_path / "ankle_leg_length_plot.pdf",
            )

    def create_morph_params_pca_scatterplot(self):
        def create_scatter_plot(x, y, c, x_label, y_label, c_label, save_path, best_x=None, best_y=None, images=None):
            # Create a figure with extended width to accommodate the images
            fig, ax = plt.subplots(figsize=(10, 6))  # Increased width to make room for images on the right

            # Create scatter plot
            scatter = ax.scatter(x=x, y=y, c=c, cmap="viridis")
            plt.colorbar(scatter, label=c_label)

            # Plot the line if best_x and best_y are provided
            if best_x is not None and best_y is not None:
                ax.plot(best_x, best_y, color="red", linewidth=2, label="Best Tensors")
                ax.legend()

                # Extend the x-axis limits to make room for the images outside the plot
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.6)  # Extend the x-axis to the right

                # Add 5 evenly spread images from the list if provided
                if images is not None:
                    num_images = min(len(best_x), len(images))  # Get the minimum between points and images
                    indices = np.round(np.linspace(0, num_images - 1, 5)).astype(int)  # 5 evenly spaced indices

                    # Calculate vertical positions for the images (evenly spaced)
                    image_y_positions = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5)

                    for idx, i in enumerate(indices):
                        x_coord = best_x.iloc[i] if hasattr(best_x, "iloc") else best_x[i]
                        y_coord = best_y.iloc[i] if hasattr(best_y, "iloc") else best_y[i]

                        # Position the image to the right, at a fixed horizontal location
                        x_offset = xlim[1] + (xlim[1] - xlim[0]) * 0.5  # Fixed offset to the right of the plot
                        y_offset = image_y_positions[idx]  # Vertically aligned based on index

                        # Place the image
                        img = OffsetImage(images[i], zoom=0.2)
                        ab = AnnotationBbox(img, (x_offset, y_offset), frameon=False, clip_on=False)
                        ax.add_artist(ab)

                        # Draw the arrow from the scatter point to the image
                        ax.annotate(
                            "",
                            xy=(x_coord, y_coord),
                            xytext=(x_offset, y_offset),
                            arrowprops=dict(color="black", arrowstyle="->")
                        )

            # Set labels and grid
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            # Close the plot to free memory
            plt.close()

        spec_folders = sorted(os.listdir(self.run_path / "specialist"), key=lambda file: get_creation_time(file, self.run_path / "specialist"))

        for i, (df, folder) in enumerate(zip(self.morph_data_dfs, spec_folders)):
            pandas_logger_df = pd.read_csv(self.run_path / "specialist" / folder / "pandas_logger_df.csv")
            folder_save_path: str = self.run_path / "specialist" / folder / "pca_plots"
            os.makedirs(folder_save_path, exist_ok=True)

            scaler: StandardScaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            pca = PCA(n_components=1)
            pca_components = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_components, columns=["PC1"])
            df_pca["Generation"] = df.index

            gen_scores = []
            for j in df.index.to_list():
                gen_scores.append(pandas_logger_df.loc[j - 1, "mean_eval"])
            df_pca["Fitness Score"] = gen_scores

            best_indices = self.best_tensors_indices[i] 
            best_tensors = df_pca.iloc[best_indices]

            create_scatter_plot(
                df_pca["Fitness Score"],
                df_pca["PC1"],
                df_pca["Generation"],
                "Fitness Score",
                "Principal Component Morphology",
                "Generations",
                folder_save_path / "one_pca_scatterplot.pdf",
                best_x=best_tensors["Fitness Score"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )

            # PCA with 2 components for the second scatterplot
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_scaled)
            df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
            df_pca["Generation"] = df.index

            gen_scores = []
            for j in df.index.to_list():
                gen_scores.append(pandas_logger_df.loc[j - 1, "mean_eval"])
            df_pca["Fitness Score"] = gen_scores

            best_tensors = df_pca.iloc[best_indices]

            create_scatter_plot(
                df_pca["PC2"],
                df_pca["PC1"],
                df_pca["Generation"],
                "2nd Principal Component Morphology",
                "1st Principal Component Morphology",
                "Generations",
                folder_save_path / "two_pca_generation_scatterplot.pdf",
                best_x=best_tensors["PC2"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )
            create_scatter_plot(
                df_pca["PC2"],
                df_pca["PC1"],
                df_pca["Fitness Score"],
                "2nd Principal Component Morphology",
                "1st Principal Component Morphology",
                "Fitness Score",
                folder_save_path / "two_pca_fitness_score_scatterplot.pdf",
                best_x=best_tensors["PC2"],
                best_y=best_tensors["PC1"],
                images=self.best_images[i]
            )

    def create_evolution_video(self):
        """Method creating evolution video by putting all images from screenshot folder back-to-back"""
        sorted_files = sorted(os.listdir(self.run_path / "specialist"), key=lambda file: get_creation_time(file, self.run_path / "specialist"))
        for folder in sorted_files:
            full_folder_path: Path = self.run_path / "specialist" / folder
            images_folder: Path = full_folder_path / "screenshots"
            images = [img for img in os.listdir(images_folder)]
            frame = cv2.imread(images_folder / images[0])
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(full_folder_path / "evolution_video.mp4", fourcc, 30, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(images_folder, image)))

            cv2.destroyAllWindows()
            video.release()

    def _evaluate_envs(self, terrains: List[TerrainType], create_videos: bool) -> List[List[Tuple[TerrainType, float]]]:
        env_fitnesses = []
        
        for terrain in terrains:
            index: int = None
            for i, terrain_list in enumerate(self.e):
                if terrain in terrain_list:
                    index = i
                    break

            fitnesses = []
            self.inds[0].setup_env_ind(self.g[index], terrain)
            for i in range(self._evaluation_count):
                fitness = self.inds[0].evaluate_fitness()
                fitnesses.append(fitness)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            env_fitnesses.append((terrain, mean_fitness))

            if create_videos:
                video_thread = threading.Thread(
                    target=self._create_video,
                    args=(terrain, copy.deepcopy(self.inds[0]), self.g[index])
                )
                video_thread.start()
        
        return env_fitnesses

    def _load_morph_data(self) -> Tuple[list[pd.DataFrame], list[list[int]], list]:
        morph_data_dfs: list[pd.DataFrame] = []
        best_tensors_indices: list[list[int]] = []
        best_images_part = []

        for folder in sorted(os.listdir(self.run_path / "specialist"), key=lambda file: get_creation_time(file, self.run_path / "specialist")):
            tensors_path = self.run_path / "specialist" / folder / "gen_tensors"
            morph_data = []
            best_tensors_index = []
            best_images = []

            sorted_tensor_files = sorted(os.listdir(tensors_path), key=lambda file: get_creation_time(file, tensors_path))

            for j, tensor_file in enumerate(sorted_tensor_files):
                if j % self.morph_step_size == 0 or tensor_file.endswith("best.pt"):
                    tensor_path = tensors_path / tensor_file
                    params = torch.load(tensor_path, weights_only=False, map_location=torch.device('cpu'))
                    self.inds[0].setup_ant_default(params)
                    
                    morph_data.append({
                        **self.inds[0].mj_env.morphology.morph_params_map,
                        "Generation": j + 1
                        })
                    if tensor_file.endswith("best.pt"):
                        best_tensors_index.append(len(morph_data) - 1)
                        best_images.append(Image.open(self.run_path / "specialist" / folder / "screenshots" / f"ant_{j+1}.png"))
            
            morph_data = pd.DataFrame(morph_data).set_index("Generation")
            morph_data_dfs.append(morph_data)
            best_tensors_indices.append(best_tensors_index)
            best_images_part.append(best_images)
        return (morph_data_dfs, best_tensors_indices, best_images_part)


class GraphBuilderCombination():
    """Class used to create graphs that combines data from multiple experimental runs."""

    def __init__(self, run_paths: list[Path] = None):
        self.ts = TrainingSchedule()
        self.evaluation_count: int = 30

        self.path_to_save: Path = Path("combined_data")

        if run_paths is not None and len(run_paths) > 0:
            self.path_to_save.mkdir(parents=True, exist_ok=True)

            # placeholder_1: Path = self._get_run_path(run_paths, "placeholder_2")
            # if placeholder_1 is not None:
            #     self.placeholder_3: pd.DataFrame = self._create_df(placeholder_1, True)
            #     self.placeholder_3.to_csv(self.path_to_save / "placeholder_2.csv", index=False)

            full_gen_morph_evo_gen_path: Path = self._get_run_path(run_paths, "FullGen-MorphEvo-Gen")
            if full_gen_morph_evo_gen_path is not None:
                save_path = self.path_to_save / "FullGen-MorphEvo-Gen.csv"
                if not save_path.exists():
                    self.full_gen_morph_evo_gen_df: pd.DataFrame = self._create_df(full_gen_morph_evo_gen_path, True)
                    self.full_gen_morph_evo_gen_df.to_csv(save_path, index=False)

            full_gen_default_morph_gen_path: Path = self._get_run_path(run_paths, "FullGen-DefaultMorph-Gen")
            if full_gen_default_morph_gen_path is not None:
                save_path = self.path_to_save / "FullGen-DefaultMorph-Gen.csv"
                if not save_path.exists():
                    self.full_gen_default_morph_gen_df: pd.DataFrame = self._create_df(full_gen_default_morph_gen_path, True)
                    self.full_gen_default_morph_gen_df.to_csv(save_path, index=False)

            our_algo_custom_morph_gen_path: Path = self._get_run_path(run_paths, "OurAlgo-CustomMorph-Gen")
            if our_algo_custom_morph_gen_path is not None:
                save_path = self.path_to_save / "OurAlgo-CustomMorph-Gen.csv"
                if not save_path.exists():
                    self.our_algo_custom_morph_gen_df: pd.DataFrame = self._create_df(our_algo_custom_morph_gen_path, True)
                    self.our_algo_custom_morph_gen_df.to_csv(save_path, index=False)

            our_algo_default_morph_gen_path: Path = self._get_run_path(run_paths, "OurAlgo-DefaultMorph-Gen")
            if our_algo_default_morph_gen_path is not None:
                save_path = self.path_to_save / "OurAlgo-DefaultMorph-Gen.csv"
                if not save_path.exists():
                    self.our_algo_default_morph_gen_df: pd.DataFrame = self._create_df(our_algo_default_morph_gen_path, True)
                    self.our_algo_default_morph_gen_df.to_csv(save_path, index=False)

            our_algo_large_morph_gen_path: Path = self._get_run_path(run_paths, "OurAlgo-LargeMorph-Gen")
            if our_algo_large_morph_gen_path is not None:
                save_path = self.path_to_save / "OurAlgo-LargeMorph-Gen.csv"
                if not save_path.exists():
                    self.our_algo_large_morph_gen_df: pd.DataFrame = self._create_df(our_algo_large_morph_gen_path, True)
                    self.our_algo_large_morph_gen_df.to_csv(save_path, index=False)

            our_algo_morph_evo_gen_path: Path = self._get_run_path(run_paths, "OurAlgo-MorphEvo-Gen")
            if our_algo_morph_evo_gen_path is not None:
                save_path = self.path_to_save / "OurAlgo-MorphEvo-Gen.csv"
                if not save_path.exists():
                    self.our_algo_morph_evo_gen_df: pd.DataFrame = self._create_df(our_algo_morph_evo_gen_path, True)
                    self.our_algo_morph_evo_gen_df.to_csv(save_path, index=False)

            our_algo_morph_evo_start_large_gen_path: Path = self._get_run_path(run_paths, "OurAlgo-MorphEvo-StartLarge-Gen")
            if our_algo_morph_evo_start_large_gen_path is not None:
                save_path = self.path_to_save / "OurAlgo-MorphEvo-StartLarge-Gen.csv"
                if not save_path.exists():
                    self.our_algo_morph_evo_start_large_gen_df: pd.DataFrame = self._create_df(our_algo_morph_evo_start_large_gen_path, True)
                    self.our_algo_morph_evo_start_large_gen_df.to_csv(save_path, index=False)

            spec_default_morph_path: Path = self._get_run_path(run_paths, "Spec-DefaultMorph")
            if spec_default_morph_path is not None:
                save_path = self.path_to_save / "Spec-DefaultMorph.csv"
                if not save_path.exists():
                    self.spec_default_morph_df: pd.DataFrame = self._create_df(spec_default_morph_path, False)
                    self.spec_default_morph_df.to_csv(save_path, index=False)

            spec_morph_evo_path: Path = self._get_run_path(run_paths, "Spec-MorphEvo")
            if spec_morph_evo_path is not None:
                save_path = self.path_to_save / "Spec-MorphEvo.csv"
                if not save_path.exists():
                    self.spec_morph_evo_df: pd.DataFrame = self._create_df(spec_morph_evo_path, False)
                    self.spec_morph_evo_df.to_csv(save_path, index=False)

            spec_morph_evo_long_path: Path = self._get_run_path(run_paths, "Spec-MorphEvo-Long")
            if spec_morph_evo_long_path is not None:
                save_path = self.path_to_save / "Spec-MorphEvo-Long.csv"
                if not save_path.exists():
                    self.spec_morph_evo_long_df: pd.DataFrame = self._create_df(spec_morph_evo_long_path, False)
                    self.spec_morph_evo_long_df.to_csv(save_path, index=False)

        else:
            self.our_algo_default_morph_gen_df: pd.DataFrame = pd.read_csv(self.path_to_save / "OurAlgo-DefaultMorph-Gen.csv")
            self.our_algo_large_morph_gen_df: pd.DataFrame = pd.read_csv(self.path_to_save / "OurAlgo-LargeMorph-Gen.csv")
            self.our_algo_morph_evo_gen_df: pd.DataFrame = pd.read_csv(self.path_to_save / "OurAlgo-MorphEvo-Gen.csv")

            self.spec_default_morph_df: pd.DataFrame = pd.read_csv(self.path_to_save / "Spec-DefaultMorph.csv") 
            self.spec_morph_evo_df: pd.DataFrame = pd.read_csv(self.path_to_save / "Spec-MorphEvo.csv")
            self.spec_morph_evo_long_df: pd.DataFrame = pd.read_csv(self.path_to_save / "Spec-MorphEvo-Long.csv")
    
    def create_graphs(self):
        self._plot_fitness_vs_environment(self.our_algo_morph_evo_gen_df, "our_algo_morph_evo_gen_fitness_env.pdf", "our_algo_morph_evo_gen: Environment fitnesses of MC-Pairs of each partition")

        self._plot_max_fitness_vs_environment(self.our_algo_morph_evo_gen_df, "our_algo_morph_evo_gen_fitness_env_ensamble_controllers.pdf", "our_algo_morph_evo_gen: Environment fitnesses of MC-Pairs ensamble")
        self._plot_max_fitness_vs_environment(self.our_algo_default_morph_gen_df, "our_algo_default_morph_gen_fitness_env_ensamble_controllers.pdf", "our_algo_default_morph_gen: Environment fitnesses of controllers ensamble")

        self._multiple_plot_max_fitness_vs_environment([self.our_algo_morph_evo_gen_df, self.spec_morph_evo_df, self.our_algo_default_morph_gen_df], ["our_algo_morph_evo_gen", "spec_morph_evo", "our_algo_default_morph_gen"], "fitness_env_experiments.pdf", "Environment fitnesses from different experiments")

        self._plot_max_fitness_boxplot_with_significance([self.our_algo_morph_evo_gen_df, self.spec_morph_evo_df, self.our_algo_default_morph_gen_df], ["our_algo_morph_evo_gen",  "spec_morph_evo", "our_algo_default_morph_gen"], "fitness_env_experiments_boxplot.pdf", "Environment fitnesses from different experiments")    

    def _get_run_path(self, run_paths, exp: str) -> Path:
        exp_paths = [path for path in run_paths if path.parts[-2] == exp]

        if len(exp_paths) == 1:
            return exp_paths[0]
        elif len(exp_paths) == 0:
            return None
        else:
            raise ValueError(f"Multiple paths with the name {exp} found.")
        
    def _create_df(self, run_path: Path, is_generalist: bool) -> pd.DataFrame:
        g = self._load_g(run_path)
        e = self._load_e(run_path)
        rows = []

        dis_morph_evo, morph_type = self._decide_ind_params(run_path)

        for i, params in enumerate(g):
            for terrain in self.ts.all_terrains:
                fitness_mean, fitness_std = self._evaluate(params, terrain, dis_morph_evo, morph_type)
                row = {
                    "Environment": terrain.short_string(),
                    "Controller": f"partition_{i}" if is_generalist else e[i][0].short_string(),
                    "Fitness": fitness_mean,
                    "Fitness std": fitness_std 
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _load_g(self, run_path: Path):
        with open(run_path / "G_var.pkl", "rb") as file:
            g = pickle.load(file)
        return g

    def _load_e(self, run_path: Path) -> List[List[TerrainType]]:
        with open(run_path / "E_var.pkl", "rb") as file:
            e = pickle.load(file)
        return e

    def _decide_ind_params(self, run_path: Path):
        '''Returns dis_morph_evo: bool and morph_type: str''' 
        if "FullGen-DefaultMorph-Gen" in str(run_path):
            return (True, "default")
        elif "FullGen-MorphEvo-Gen" in str(run_path):
            return (False, None)
        elif "OurAlgo-CustomMorph-Gen" in str(run_path):
            return (True, "custom")
        elif "OurAlgo-DefaultMorph-Gen" in str(run_path):
            return (True, "default")
        elif "OurAlgo-LargeMorph-Gen" in str(run_path):
            return (True, "large")
        elif "OurAlgo-MorphEvo-Gen" in str(run_path):
            return (False, None)
        elif "OurAlgo-MorphEvo-StartLarge-Gen" in str(run_path):
            return (False, None)
        elif "Spec-DefaultMorph" in str(run_path):
            return (True, "default")
        elif "Spec-MorphEvo" in str(run_path):
            return (False, None)
        elif "Spec-MorphEvo-Long" in str(run_path):
            return (False, None)

    def _evaluate(self, params: Tensor, terrain: TerrainType, dis_morph_evo: bool, morph_type: str) -> Tuple[float, float]:
        parallel_jobs = 30
        fitnesses = []
        inds: list[Individual] = [
            Individual(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                Algo.morph_params_bounds_enc,
                Algo.penalty_growth_rate,
                Algo.penalty_scale_factor,
                Algo.penalty_scale_factor_err,
                dis_morph_evo,
                morph_type)
            for _ in range(parallel_jobs)
        ]
        for ind in inds: ind.setup_env_ind(params, terrain)

        def eval(ind: Individual) -> float: return ind.evaluate_fitness() 
        for _ in range(0, self.evaluation_count, parallel_jobs):
            batch_size = min(parallel_jobs, self.evaluation_count - len(fitnesses))

            tasks = (joblib.delayed(eval)(ind) for ind in inds[:batch_size])
            batch_fitness = joblib.Parallel(n_jobs=parallel_jobs)(tasks)
            fitnesses.extend(batch_fitness)

        fitnesses = np.array(fitnesses)
        return (np.mean(fitnesses), np.std(fitnesses))

    def _plot_fitness_vs_environment(self, data: pd.DataFrame, save_as_name: str, title: str):
        categories = data["Environment"].unique()
        data["Environment"] = pd.Categorical(data["Environment"], categories=categories, ordered=True)
        
        num_envs = len(categories)
        
        fig_width = max(20, num_envs * 0.2)
        fig_height = 4 
        
        plt.figure(figsize=(fig_width, fig_height))

        sns.lineplot(
            x="Environment",
            y="Fitness",
            hue="Controller",
            data=data,
            marker="o",
            errorbar=None,
            palette="tab10"
        )

        plt.title(title, fontsize=16)
        plt.xlabel("Environment", fontsize=14)
        plt.ylabel("Fitness", fontsize=14)
        plt.legend(title="MC-Pair")
        
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        
        plt.savefig(self.path_to_save / save_as_name, bbox_inches="tight")
        plt.close() 

    def _plot_max_fitness_vs_environment(self, data: pd.DataFrame, save_as_name: str, title: str):
        max_fitness = data.groupby("Environment", as_index=False, observed=False)["Fitness"].max()

        categories = data["Environment"].unique()
        max_fitness["Environment"] = pd.Categorical(max_fitness["Environment"], categories=categories, ordered=True)
        
        num_envs = len(categories)
        
        fig_width = max(20, num_envs * 0.2)
        fig_height = 4

        plt.figure(figsize=(fig_width, fig_height))

        sns.lineplot(
            x="Environment",
            y="Fitness",
            data=max_fitness,
            marker="o",
            color="red",
            label="Max Fitness",
            errorbar=None,
            legend=False
        )
        
        plt.title(title, fontsize=16)
        plt.xlabel("Environment", fontsize=14)
        plt.ylabel("Fitness", fontsize=14)
        
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        
        plt.savefig(self.path_to_save / save_as_name, bbox_inches="tight")
        plt.close()

    def _multiple_plot_max_fitness_vs_environment(self, dataframes: list, labels: list, save_as_name: str, title: str):
        plt.figure(figsize=(20, 4))
        
        for data, label in zip(dataframes, labels):
            max_fitness = data.groupby("Environment", as_index=False, observed=False)["Fitness"].max()
            
            categories = data["Environment"].unique()
            max_fitness["Environment"] = pd.Categorical(max_fitness["Environment"], categories=categories, ordered=True)

            sns.lineplot(
                x="Environment",
                y="Fitness",
                data=max_fitness,
                marker="o",
                label=label,
                errorbar=None
            )

        plt.title(title, fontsize=16)
        plt.xlabel("Environment", fontsize=14)
        plt.ylabel("Fitness", fontsize=14)
        
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        
        plt.savefig(self.path_to_save / save_as_name, bbox_inches="tight")
        plt.close()

    def _plot_max_fitness_boxplot_with_significance(self, dataframes, labels, output_file, title):
        combined_df = pd.concat(
            [df.assign(Experiment=label) for df, label in zip(dataframes, labels)],
            ignore_index=True
        )

        max_fitness_df = combined_df.groupby(["Experiment", "Environment"])["Fitness"].max().reset_index()

        sns.set(style="whitegrid")

        palette = sns.color_palette("Set2", len(labels))

        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.boxplot(
            x="Experiment", 
            y="Fitness", 
            data=max_fitness_df, 
            hue="Experiment", 
            palette=palette, 
            boxprops={"alpha": 0.2},
            width=0.5,
            legend=False
        )

        sns.stripplot(
            x="Experiment", 
            y="Fitness", 
            data=max_fitness_df, 
            hue="Experiment", 
            palette=palette,
            alpha=0.9, 
            jitter=True,
            dodge=False,
            ax=ax,
            legend=False
        )

        pairs = list(itertools.combinations(labels, 2))
        y_max = max_fitness_df["Fitness"].max()
        y_min = max_fitness_df["Fitness"].min()
        y_range = y_max - y_min
        y_offset = y_range * 0.05

        for i, (exp1, exp2) in enumerate(pairs):
            data1 = max_fitness_df[max_fitness_df["Experiment"] == exp1]["Fitness"]
            data2 = max_fitness_df[max_fitness_df["Experiment"] == exp2]["Fitness"]
            
            stat, p_val = kruskal(data1, data2)

            if p_val < 0.0001:
                significance = "****"
            elif p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            else:
                significance = "ns"

            x1, x2 = labels.index(exp1), labels.index(exp2)
            y = max(max_fitness_df["Fitness"][max_fitness_df["Experiment"] == exp1].max(),
                    max_fitness_df["Fitness"][max_fitness_df["Experiment"] == exp2].max()) + y_offset

            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, color="k")

            ax.text((x1 + x2) * 0.5, y + y_offset, significance, ha="center", va="bottom", color="k", fontsize=12)

        ax.set_title(title, fontsize=16, weight="bold", pad=15)
        ax.set_xlabel("Experiment", fontsize=14, weight="bold")
        ax.set_ylabel("Fitness Score", fontsize=14, weight="bold")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
        sns.despine(trim=True)

        plt.tight_layout()
        plt.savefig(self.path_to_save / output_file, bbox_inches="tight", dpi=300)
        plt.close()


def get_creation_time(file, path):
    return os.path.getctime(path / file)