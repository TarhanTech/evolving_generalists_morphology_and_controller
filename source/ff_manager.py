"""This module defines classes for managing folder creation and file saving operations."""

from abc import ABC
from pathlib import Path
import json
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
        date_time = datetime.now().strftime("%d-%m_%H-%M-%S-%f")
        self.root_folder: Path = Path(f"./runs/{subfolder}_{date_time}")
        self.root_folder.mkdir(parents=True, exist_ok=True)

    def save_pickle(self, filename: str, data: any):
        """Saves data to a pickle"""
        pickle_path: Path = self.root_folder / filename
        with open(pickle_path, "wb") as file:
            pickle.dump(data, file)

    def log_evaluations(self, regarding: str, number_of_evals: int):
        eval_log_path: Path = self.root_folder / "number_of_evals.log"
        with open(eval_log_path, "a") as file:
            file.write(f"{regarding} number of evals: {number_of_evals}\n")


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

    def save_pandas_logger_df(self, number: int, pandas_logger: PandasLogger):
        """Saves the DataFrame of the pandas logger of evotorch containing mean, best and median scores to a CSV file."""
        df = pandas_logger.to_dataframe()
        pd.DataFrame(df).to_csv(self.root_folder / f"partition_{number}" / "pandas_logger_df.csv", index=False)

    def save_df(self, number: int, df: pd.DataFrame, file_name: str):
        """Saves the DataFrame. Filename must end with .csv"""
        pd.DataFrame(df).to_csv(self.root_folder / f"partition_{number}" / file_name, index=False)

    def save_json(self, number: int, dict, file_name: str):
        with open(self.root_folder / f"partition_{number}" / file_name, 'w') as file:
            json.dump(dict, file)

class FFManagerSpecialist(FFManager):
    """Manages folder creation and file saving specifically for specialist runs."""

    def __init__(self, subfolder: str):
        super().__init__(subfolder)
        self.specialist_folder = self.root_folder / "specialist"
        self.specialist_folder.mkdir(parents=True, exist_ok=True)

    def create_terrain_folder(self, terrain: TerrainType):
        """Creates folders for a specific partition, including 'screenshots' and 'gen_tensors'."""
        path_to_save: Path = self._get_path_to_save(terrain)
        os.makedirs(path_to_save / "screenshots", exist_ok=True)
        os.makedirs(path_to_save / "gen_tensors", exist_ok=True)

    def save_screenshot_ant(self, terrain: TerrainType, gen: int, pop_best: torch.Tensor, ind: Individual):
        """Saves a screenshot of the ant."""
        ind.setup_ant_default(pop_best)
        ind.make_screenshot_ant(self._get_path_to_save(terrain) / "screenshots" / f"ant_{gen}.png")

    def save_specialist_tensor(self, terrain: TerrainType, gen: int, params: torch.Tensor, new_best: bool):
        """Saves a tensor representing the parameters of the generalist model."""
        tensor_filename = f"tensor_{gen}_best.pt" if new_best else f"tensor_{gen}.pt"
        torch.save(params, self._get_path_to_save(terrain) / "gen_tensors" / tensor_filename)

    def save_pandas_logger_df(self, terrain: TerrainType, pandas_logger: PandasLogger):
        """Saves the DataFrame of the pandas logger of evotorch containing mean, best and median scores to a CSV file."""
        df = pandas_logger.to_dataframe()
        pd.DataFrame(df).to_csv(self._get_path_to_save(terrain) / "pandas_logger_df.csv", index=False)

    def _get_path_to_save(self, terrain: TerrainType):
        path_to_save: Path = self.specialist_folder / terrain.__str__()
        return path_to_save
    