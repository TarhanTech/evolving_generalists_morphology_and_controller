import pandas as pd
import torch
import pickle
from pathlib import Path
from torch import Tensor
from source.individual import Individual

def get_max_fitness_in_partition(partition_folder: Path) -> float:
    df = pd.read_csv(partition_folder / "gen_score_pandas_df.csv")
    return df["Generalist Score"].max()

def main():
    exp_path = Path('./runs/OurAlgo-MorphEvo-Gen')
    ind = Individual("cpu", (-0.1, 0.1), 1.03, 100, 1000, False)
    data_list = []
    for run_folder in sorted(exp_path.iterdir(), key=lambda p: p.name):


if __name__ == "__main__":
    main()