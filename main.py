"""
Execute this file to run one of the experiments described in the paper.
Example: 'nohup python main.py experiment2 > experiment2.log 2>&1 &'
"""

from typing import List, Union
from pathlib import Path
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from source.individual import Individual
from source.training_env import DefaultTerrain, HillsTerrain, RoughTerrain, TerrainType
from source.algo import Experiment1, Experiment2, Experiment3, Experiment4, Experiment5


def create_plot(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["pop_best_eval"], label="Population Best Evaluation", marker="s")
    plt.xlabel("Generation")
    plt.ylabel("Evaluation Values")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True)
    # Save the plot as a file
    plt.savefig(f"{save_path}/evaluation_metrics_plot.png", dpi=300, bbox_inches="tight")


def experiment1():
    """Runs experiment 1"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment1 = Experiment1(23)
    algo.run()
    print("Experiment 1 has finished!!")


def experiment2():
    """Runs experiment 2"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment2 = Experiment2(23)
    algo.run()
    print("Experiment 2 has finished!!")



def experiment3():
    """Runs experiment 3"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment3 = Experiment3(23)
    algo.run()
    print("Experiment 3 has finished!!")


def experiment4():
    """Runs experiment 3"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment4 = Experiment4(23)
    algo.run()
    print("Experiment 4 has finished!!")


def experiment5():
    """Runs experiment 3"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment5 = Experiment5(23)
    algo.run()
    print("Experiment 5 has finished!!")


def test_ant(tensor_path: Path, terrain: str, params: str):
    """Runs a test"""

    def convert_param(param: str) -> Union[int, float]:
        """Convert a string parameter to int or float."""
        try:
            return int(param)
        except ValueError:
            try:
                return float(param)
            except ValueError as e:
                raise ValueError(f"Parameter '{param}' is not a valid integer or float.") from e

    terrain_type: TerrainType = TERRAIN_MAP[terrain]

    ind: Individual = Individual(id=99)
    tensor_params = torch.load(tensor_path)
    if terrain_type is DefaultTerrain:
        ind.setup_ant_default(tensor_params)
    elif terrain_type is HillsTerrain:
        params: List[Union[int, float]] = [convert_param(p) for p in params]
        ind.setup_ant_hills(tensor_params, params[0], params[1])
    elif terrain_type is RoughTerrain:
        params: List[Union[int, float]] = [convert_param(p) for p in params]
        ind.setup_ant_rough(tensor_params, params[0], params[1])
    else:
        assert False, "Class type not supported"

    total_reward: float = ind.evaluate_fitness(render_mode="human")
    print(f"Total Rewards: {total_reward}")


TERRAIN_MAP = {
    "default": DefaultTerrain,
    "hills": HillsTerrain,
    "rough": RoughTerrain,
}


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="experiment", required=True, help="Choose an experiment to run.")

    parser_exp1 = subparsers.add_parser(
        "experiment1", help="Run the first experiment where you create a generalist for each partition of the environments."
    )

    parser_exp2 = subparsers.add_parser(
        "experiment2",
        help="Run the second experiment where you create one generalist for all the environments",
    )

    parser_exp3 = subparsers.add_parser("experiment3", help="run the third experiment where you create a specialist for each of the environments")

    parser_exp4 = subparsers.add_parser(
        "experiment4",
        help="Run the fourth experiment where you create a specialist for each of the environments using same resources as experiment 1",
    )

    parser_exp5 = subparsers.add_parser(
        "experiment5",
        help="Run the fifth experiment where you create a generalist (no morphological evolution) for each partition of the environments.",
    )

    parser_test = subparsers.add_parser("test", help="Run a test experiment to visual a generated tensor from the tensor folders")
    parser_test.add_argument("--tensor", type=Path, required=True, help="Path to the tensor that you want to test.")
    parser_test.add_argument("--terrain", type=str, choices=TERRAIN_MAP.keys(), required=True, help="Terrain environment to test in")
    parser_test.add_argument(
        "--params",
        type=str,
        nargs="+",
        required=False,
        help="A list of parameters for the selected class. Parameters can be integers or floats.",
    )

    args = parser.parse_args()

    if args.experiment == "experiment1":
        experiment1()
    elif args.experiment == "experiment2":
        experiment2()
    elif args.experiment == "experiment3":
        experiment3()
    elif args.experiment == "experiment4":
        experiment4()
    elif args.experiment == "experiment5":
        experiment5()
    elif args.experiment == "test":
        test_ant(args.tensor, args.terrain, args.params)
    else:
        print("Something went wrong!")


if __name__ == "__main__":
    main()
