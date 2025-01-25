import argparse
from pathlib import Path
import pickle
import pandas as pd
from typing import List, Tuple
from source.training_env import TerrainType


def calculate_mean_fitnesses(file_path: Path) -> float:
    if not file_path.exists():
        print(f"Warning: File '{file_path}' not found.")
        return None

    with open(file_path, 'rb') as file:
        data: List[Tuple[TerrainType, float]] = pickle.load(file)

    return sum(fitness for _, fitness in data) / len(data) if data else 0.0


def main():
    parser = argparse.ArgumentParser(description="Iterate over folders in a specified directory.")
    parser.add_argument(
        "--exp_path",
        type=Path,
        required=True,
        help="The path to the directory containing run folders."
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Optional path to save the output CSV. Defaults to the current working directory."
    )
    args = parser.parse_args()

    exp_path: Path = args.exp_path
    out_path: Path = args.out_path if args.out_path else Path.cwd()

    if not exp_path.exists():
        print(f"Error: The path '{exp_path}' does not exist.")
        return

    if not exp_path.is_dir():
        print(f"Error: The path '{exp_path}' is not a directory.")
        return

    data_for_df = []

    for run_folder in exp_path.iterdir():
        if run_folder.is_dir():
            test_data_path = run_folder / "env_fitnesses_test.pkl"
            train_data_path = run_folder / "env_fitnesses_training.pkl"

            test_mean = calculate_mean_fitnesses(test_data_path)
            train_mean = calculate_mean_fitnesses(train_data_path)

            data_for_df.append({
                "fitness_test": test_mean,
                "fitness_training": train_mean
            })
        else:
            print(f"Skipping non-folder item: {run_folder}")

    df = pd.DataFrame(data_for_df, columns=["fitness_test", "fitness_training"])

    csv_output_path = out_path / f"{args.exp_path.name}_fitness_data.csv"
    df.to_csv(csv_output_path, index=False)

    print(f"Saved DataFrame to: {csv_output_path}")


if __name__ == "__main__":
    main()
