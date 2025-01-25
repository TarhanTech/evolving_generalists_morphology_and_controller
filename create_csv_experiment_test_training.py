import argparse
from pathlib import Path
import pickle
import pandas as pd
from typing import List, Tuple
from source.training_env import TerrainType

def load_fitnesses(file_path: Path, split_name: str) -> pd.DataFrame:
    if not file_path.exists():
        print(f"Warning: File '{file_path}' not found.")
        return pd.DataFrame(columns=["terrain_type", "fitness", "split"])

    with open(file_path, 'rb') as file:
        data: List[Tuple[TerrainType, float]] = pickle.load(file)

    records = []
    for terrain_type, fitness_val in data:
        records.append({
            "fitness": fitness_val,
            "split": split_name  # either "test" or "train"
        })

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Load test and training fitnesses from a single run folder.")
    parser.add_argument(
        "--run_folder",
        type=Path,
        required=True,
        help="The path to the run folder (containing env_fitnesses_test.pkl and env_fitnesses_training.pkl)."
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Path to save the output CSV. Defaults to the current working directory."
    )
    args = parser.parse_args()

    run_folder: Path = args.run_folder
    out_path: Path = args.out_path if args.out_path else Path.cwd()

    if not run_folder.exists():
        print(f"Error: The folder '{run_folder}' does not exist.")
        return
    if not run_folder.is_dir():
        print(f"Error: '{run_folder}' is not a directory.")
        return

    test_data_path = run_folder / "env_fitnesses_test.pkl"
    train_data_path = run_folder / "env_fitnesses_training.pkl"

    df_test = load_fitnesses(test_data_path, split_name="test")
    df_train = load_fitnesses(train_data_path, split_name="train")

    df = pd.concat([df_test, df_train], ignore_index=True)

    csv_output_path = out_path / f"{run_folder.name}_fitness_details.csv"
    df.to_csv(csv_output_path, index=False)

    print(f"Saved terrain-by-terrain fitness data to: {csv_output_path}")


if __name__ == "__main__":
    main()
