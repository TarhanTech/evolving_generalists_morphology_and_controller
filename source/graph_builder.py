"""This module contains graphbuilder classes"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

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
from scipy.stats import mode
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

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

    def __init__(self, run_path: Path, create_videos: bool, dis_morph_evo):
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
                dis_morph_evo
            )
            for _ in range(10)
        ]
        
        self.ts: TrainingSchedule = TrainingSchedule()

        self.g: List[Tensor] = self._load_g()
        self.e: List[List[TerrainType]] = self._load_e()
        self._print_run_data()

        self._evaluation_count: int = 50
        
    @abstractmethod
    def create_ant_screenshots(self):
        """Method that creates photos of the ants morphology in the environment"""
        pass

    def create_generalist_heatmap_partition(self):
        """Method that creates heatmap to show which environments are being handled by which partition"""
        default_df, hills_df, rt_df = self._create_dataframe_terrains()

        for j in range(len(self.e)):
            for env in self.e[j]:
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
            self.run_path / "generalist_heatmap_partition.pdf",
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

    def _load_e(self):
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
        def setup_env_ind(terrain: TerrainType, params: Tensor):
            if isinstance(terrain, RoughTerrain):
                for ind in self.inds: ind.setup_ant_rough(params, terrain.floor_height, terrain.block_size)
            elif isinstance(terrain, HillsTerrain):
                for ind in self.inds: ind.setup_ant_hills(params, terrain.floor_height, terrain.scale)
            elif isinstance(terrain, DefaultTerrain):
                for ind in self.inds: ind.setup_ant_default(params)
            else:
                assert False, "Class type not supported"

        def eval(ind: Individual):
            return ind.evaluate_fitness()

        if terrain in self.ts.training_terrains:
            params: Tensor = None
            for i, terrains in enumerate(self.e):
                if terrain in terrains:
                    params = self.g[i]
                    break

            if create_videos:
                video_thread = threading.Thread(
                    target=self._create_video,
                    args=(terrain, copy.deepcopy(self.inds[0]), params)
                )
                video_thread.start()

            setup_env_ind(terrain, params)

            fitnesses: list[float] = []
            batch_size = len(self.inds)
            for i in range(0, self._evaluation_count, batch_size):
                tasks = (joblib.delayed(eval)(ind) for ind in self.inds)
                batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
                fitnesses.extend(batch_fitness)

            mean_fitness = sum(fitnesses) / len(fitnesses)
            return (terrain, mean_fitness)
        elif terrain in self.ts.testing_terrains:
            fitnesses_part: list[float] = []
            for params in self.g:
                setup_env_ind(terrain, params)
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
                    args=(terrain, copy.deepcopy(self.inds[0]), params)
                )
                video_thread.start()

            return (terrain, highest_fitness)
        else: 
            raise ValueError(f"Terrain {terrain} not found in training or testing terrains.")

    def _create_video(self, terrain, ind: Individual, params):
        if isinstance(terrain, RoughTerrain):
            ind.setup_ant_rough(params, terrain.floor_height, terrain.block_size)
        elif isinstance(terrain, HillsTerrain):
            ind.setup_ant_hills(params, terrain.floor_height, terrain.scale)
        elif isinstance(terrain, DefaultTerrain):
            ind.setup_ant_default(params)
        else:
            assert False, "Class type not supported"

        video_save_path = self.run_path / "videos_env" / terrain.__str__()
        ind.evaluate_fitness(render_mode="rgb_array", video_save_path=video_save_path)

    def _change_folder_name(self):
        fitness_only = np.array([x[1] for x in self.env_fitnesses])
        mean_fitness = round(np.mean(fitness_only))

        current_run_path = self.run_path

        folder_name = current_run_path.name

        pattern = r'^(exp\d+_(?:gen|spec)_)(\d+_)?(.*)$'

        def repl(match):
            prefix = match.group(1)  # 'expX_gen_' or 'expX_spec_'
            suffix = match.group(3)  # The rest of the folder name
            return f'{prefix}{mean_fitness}_{suffix}'

        # Apply the substitution to get the new folder name
        new_folder_name = re.sub(pattern, repl, folder_name)

        # Construct the new run path
        new_run_path = current_run_path.parent / new_folder_name

        # Rename the folder (ensure the new path doesn't already exist)
        if not new_run_path.exists():
            current_run_path.rename(new_run_path)
            self.run_path = new_run_path
        else:
            print(f"Cannot rename: {new_run_path} already exists.")
        

class GraphBuilderGeneralist(Graphbuilder):
    """Class used to create graphs, images and videos for the experimental runs dedicated for generalist runs"""

    def __init__(self, run_path: Path, create_videos: bool = False, dis_morph_evo = False):
        super().__init__(run_path, create_videos, dis_morph_evo)
        self.env_fitnesses_test: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.testing_terrains, create_videos)
        self.env_fitnesses_training: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.training_terrains, create_videos)
        self.env_fitnesses: List[Tuple[TerrainType, float]] = self.env_fitnesses_test + self.env_fitnesses_training

        self._change_folder_name()
        self._print_run_data()
        
        self.morph_step_size = 10 
        
        if dis_morph_evo is False:
            x1, x2, x3 = self._load_morph_data()
            self.morph_data_dfs: list[pd.DataFrame] = x1
            self.best_tensors_indices: list[list[int]] = x2
            self.best_images = x3
        
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
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            num_generations: int = 0
            for _, values in data.items():
                num_generations += len(values)


            fitness_evals_envs_path = self.run_path / f"partition_{i+1}" / "fitness_evals_envs"
            fitness_evals_envs_path.mkdir(parents=True, exist_ok=True)
            for key, values in data.items():
                original_indices = np.linspace(0, num_generations, num=len(values))

                interpolator = interp1d(original_indices, values, kind='linear')
                new_indices = np.linspace(0, num_generations, num=num_generations)
                interpolated_values = interpolator(new_indices)
                plt.figure()
                plt.plot(new_indices, interpolated_values)
                plt.title(f'{key} Plot')
                plt.xlabel('Generations')
                plt.ylabel('Fitness')
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
                ax.plot(best_x, best_y, color='red', linewidth=2, label="Best Tensors")
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
                        x_coord = best_x.iloc[i] if hasattr(best_x, 'iloc') else best_x[i]
                        y_coord = best_y.iloc[i] if hasattr(best_y, 'iloc') else best_y[i]

                        # Position the image to the right, at a fixed horizontal location
                        x_offset = xlim[1] + (xlim[1] - xlim[0]) * 0.5  # Fixed offset to the right of the plot
                        y_offset = image_y_positions[idx]  # Vertically aligned based on index

                        # Place the image
                        img = OffsetImage(images[i], zoom=0.2)
                        ab = AnnotationBbox(img, (x_offset, y_offset), frameon=False, clip_on=False)
                        ax.add_artist(ab)

                        # Draw the arrow from the scatter point to the image
                        ax.annotate(
                            '',
                            xy=(x_coord, y_coord),
                            xytext=(x_offset, y_offset),
                            arrowprops=dict(color='black', arrowstyle='->')
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
        def get_creation_time(image_file):
            return os.path.getctime(images_folder / image_file)
        
        for i in range(len(self.g)):
            partition_folder: Path = self.run_path / f"partition_{i+1}"
            images_folder: Path = partition_folder / "screenshots"
            sorted_image_files = sorted(os.listdir(images_folder), key=get_creation_time)

            frame = cv2.imread(str(images_folder / sorted_image_files[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(partition_folder / "evolution_video.mp4", fourcc, 30, (width, height))

            for image in sorted_image_files:
                video.write(cv2.imread(os.path.join(images_folder, image)))

            cv2.destroyAllWindows()
            video.release()
        
    def _load_morph_data(self) -> Tuple[list[pd.DataFrame], list[list[int]], list]:
        def get_creation_time(tensor_file):
            return os.path.getctime(tensors_path / tensor_file)
        
        morph_data_dfs: list[pd.DataFrame] = []
        best_tensors_indices: list[list[int]] = []
        best_images_part = []

        for i, _ in enumerate(self.g):
            tensors_path = self.run_path / f"partition_{i+1}" / "gen_tensors"
            morph_data = []
            best_tensors_index = []
            best_images = []

            sorted_tensor_files = sorted(os.listdir(tensors_path), key=get_creation_time)

            for j, tensor_file in enumerate(sorted_tensor_files):
                if j % self.morph_step_size == 0 or tensor_file.endswith("best.pt"):
                    tensor_path = tensors_path / tensor_file
                    params = torch.load(tensor_path)
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

# TODO: Fix this class same way generalist was fixed
class GraphBuilderSpecialist(Graphbuilder):
    """Class used to create graphs, images and videos for the experimental runs dedicated for specialist runs"""

    def __init__(self, run_path: Path, create_videos: bool = False):
        super().__init__(run_path, create_videos)

        self.ts.training_terrains = self.ts.all_terrains
        self.ts.testing_terrains = []

        self.env_fitnesses_test: List[Tuple[TerrainType, float]] = []
        self.env_fitnesses_training: List[Tuple[TerrainType, float]] = self._evaluate_envs(self.ts.training_terrains, create_videos)
        self.env_fitnesses: List[Tuple[TerrainType, float]] = self.env_fitnesses_test + self.env_fitnesses_training

        self._change_folder_name()
        self._print_run_data()

        self.morph_data_dfs: list[pd.DataFrame] = self._load_morph_data()

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

        for df, folder in zip(self.morph_data_dfs, os.listdir(self.run_path / "specialists")):
            folder_save_path: Path = self.run_path / "specialists" / folder / "morph_params_evolution_plots"
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
        for df, folder in zip(self.morph_data_dfs, os.listdir(self.run_path / "specialists")):
            folder_save_path: str = self.run_path / "specialists" / folder
            scaler: StandardScaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(df_scaled)

            df_pca = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
            df_pca["generations"] = range(1, len(df_pca) + 1)

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
                folder_save_path / "pca_scatterplot.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def create_evolution_video(self):
        """Method creating evolution video by putting all images from screenshot folder back-to-back"""
        for folder in os.listdir(self.run_path / "specialist"):
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

    def _load_morph_data(self) -> list[list[pd.DataFrame]]:
        morph_data_dfs: list[pd.DataFrame] = []

        for folder in os.listdir(self.run_path / "specialist"):
            full_path = self.run_path / "specialist" / folder

            if full_path.is_dir():
                tensors_path: Path = full_path / "gen_tensors"
                morph_data = []

                for tensor_file in os.listdir(tensors_path):
                    tensor_path = tensors_path / tensor_file
                    params = torch.load(tensor_path)
                    self.inds[0].setup_ant_default(params)
                    morph_data.append(self.inds[0].mj_env.morphology.morph_params_map)
                morph_data_dfs.append(pd.DataFrame(morph_data))
        return morph_data_dfs

# TODO: When all experiments are established in algo.py. Finish this class
class GraphBuilderCombination:
    """Class used to create graphs that combines data from multiple experimental runs."""

    def __init__(self, gbs: List[Graphbuilder]):
        self.gbs: List[Graphbuilder] = gbs

    def create_boxplot_experiments(self):
        """Method that creates a graph with boxplots of different experiments"""
        if len(self.gbs) <= 1:
            return

        labels: List[str] = []
        fitness_values: List[float] = []
        for i, gb in enumerate(self.gbs):
            fitness_only: list[float] = [x[1] for x in gb.env_fitnesses]
            label: str = gb.run_path.name
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

        plt.savefig("./fitness_boxplot_experiments.pdf", dpi=300, bbox_inches="tight")
        plt.close()
