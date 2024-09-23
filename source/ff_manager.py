"""This module defines classes for managing folder creation and file saving operations."""

from abc import ABC
from pathlib import Path
import os
import pickle
from datetime import datetime
from typing import List
import pandas as pd
import torch
from evotorch.logging import PandasLogger
from source.individual import Individual
from source.training_env import (
    DefaultTerrain,
    RoughTerrain,
    HillsTerrain,
    TerrainType,
)

class FFManager(ABC):
    """Base class for managing folder creation and file saving operations."""

    def __init__(self, subfolder: str):
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.root_folder: Path = Path(f"./runs/{subfolder}_{date_time}")
        self.root_folder.mkdir(parents=True, exist_ok=True)

    def save_pickle(self, filename: str, data: any):
        """Saves data to a pickle"""
        pickle_path: Path = self.root_folder / filename
        with open(pickle_path, "wb") as file:
            pickle.dump(data, file)


class FFManagerGeneralist(FFManager):
    """Manages folder creation and file saving specifically for generalist runs."""

    def __init__(self, subfolder: str):
        super().__init__(subfolder)

    def create_partition_folder(self, number: int):
        """Creates folders for a specific partition, including 'screenshots' and 'gen_tensors'."""
        os.makedirs(self.root_folder / f"partition_{number}" / "screenshots", exist_ok=True)
        os.makedirs(self.root_folder / f"partition_{number}" / "gen_tensors", exist_ok=True)

    def save_screenshot_ant(self, number: int, gen: int, pop_best: torch.Tensor, ind: Individual):
        """Saves a screenshot of the ant."""
        ind.setup_ant_default(pop_best)
        ind.make_screenshot_ant(self.root_folder / f"partition_{number}" / "screenshots" / f"ant_{gen}.png")

    def save_generalist_tensor(self, number: int, gen: int, params: torch.Tensor, new_best: bool):
        """Saves a tensor representing the parameters of the generalist model."""
        tensor_filename = f"tensor_{gen}_best.pt" if new_best else f"tensor_{gen}.pt"
        torch.save(params, self.root_folder / f"partition_{number}" / f"gen_tensors/{tensor_filename}")

    def save_generalist_score_df(self, number: int, df_gen_scores: pd.DataFrame):
        """Saves the DataFrame containing generalist scores to a CSV file."""
        pd.DataFrame(df_gen_scores).to_csv(self.root_folder / f"partition_{number}" / "gen_score_pandas_df.csv", index=False)

    def save_pandas_logger_df(self, number: int, pandas_logger: PandasLogger):
        """Saves the DataFrame of the pandas logger of evotorch containing mean, best and median scores to a CSV file."""
        df = pandas_logger.to_dataframe()
        pd.DataFrame(df).to_csv(self.root_folder / f"partition_{number}" / "pandas_logger_df.csv", index=False)

class FFManagerSpecialist(FFManager):
    """Manages folder creation and file saving specifically for specialist runs."""

    def __init__(self, subfolder: str):
        super().__init__(subfolder)

    def create_terrain_folder(self, terrain: TerrainType):
        """Creates folders for a specific partition, including 'screenshots' and 'gen_tensors'."""
        path_to_save: Path = self._get_path_to_save(terrain)
        os.makedirs(path_to_save / "screenshots", exist_ok=True)
        os.makedirs(path_to_save / "gen_tensors", exist_ok=True)

    def save_screenshot_ant(self, terrain: TerrainType, gen: int, pop_best: torch.Tensor, ind: Individual):
        """Saves a screenshot of the ant."""
        ind.setup_ant_default(pop_best)
        ind.make_screenshot_ant(self._get_path_to_save(terrain) / "screenshots" / f"ant_{gen}.png")

    def save_specialist_tensor(self, terrain: TerrainType, gen: int, params: torch.Tensor):
        """Saves a tensor representing the parameters of the generalist model."""
        torch.save(params, self._get_path_to_save(terrain) / "gen_tensors" / f"tensor_{gen}.pt")

    def save_pandas_logger_df(self, terrain: TerrainType, pandas_logger: PandasLogger):
        """Saves the DataFrame of the pandas logger of evotorch containing mean, best and median scores to a CSV file."""
        df = pandas_logger.to_dataframe()
        pd.DataFrame(df).to_csv(self._get_path_to_save(terrain) / "pandas_logger_df.csv", index=False)

    def _get_path_to_save(self, terrain: TerrainType):
        path_to_save: Path = None
        if isinstance(terrain, RoughTerrain):
            path_to_save = self.root_folder / f"{type(terrain).__name__}_{terrain.block_size}_{terrain.floor_height}"
        elif isinstance(terrain, HillsTerrain):
            path_to_save = self.root_folder / f"{type(terrain).__name__}_{terrain.scale}_{terrain.floor_height}"
        elif isinstance(terrain, DefaultTerrain):
            path_to_save = self.root_folder / f"{type(terrain).__name__}"
        else:
            raise ValueError(f"Instance of 'terrain' is not a supported type.")
        
        return path_to_save
    