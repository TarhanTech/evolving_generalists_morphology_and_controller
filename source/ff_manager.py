"""This module defines classes for managing folder creation and file saving operations."""

from abc import ABC
from pathlib import Path
import time
import os
import pickle
from typing import List
import pandas as pd
import torch
from source.individual import Individual
from source.training_env import TerrainType

class FFManager(ABC):
    """Base class for managing folder creation and file saving operations."""

    def __init__(self, root_folder: Path):
        self.root_folder: Path = root_folder
        os.makedirs(self.root_folder, exist_ok=True)


class FFManagerGeneralist(FFManager):
    """Manages folder creation and file saving specifically for generalist runs."""

    def __init__(self):
        super().__init__(Path(f"runs/run_generalist_{time.time()}"))

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

    def save_generalist_score_df(self, df_gen_scores: pd.DataFrame):
        """Saves the DataFrame containing generalist scores to a CSV file."""
        pd.DataFrame(df_gen_scores).to_csv(self.root_folder / "gen_score_pandas_df.csv", index=False)

    def save_generalists(self, g: List[torch.Tensor]):
        """Saves a list of generalist tensors to a pickle file."""
        with open(self.root_folder / "G_var.pkl", "wb") as file:
            pickle.dump(g, file)

    def save_environments(self, e: List[List[TerrainType]]):
        """Saves a list of environments to a pickle file."""
        with open(self.root_folder / "E_var.pkl", "wb") as file:
            pickle.dump(e, file)
