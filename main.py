"""Execute this file to run one of the experiments described in the paper."""

from typing import List, Tuple
import time
from pathlib import Path
import pickle
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger
from source.individual import Individual
from source.ant_problem import AntProblem
from source.training_env import DefaultTerrain, HillsTerrain, RoughTerrain, TerrainType
from source.algo import Experiment1

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

def train_specialist_ants():
    parallel_jobs: int = 6
    if not os.path.exists(train_ant_xml_folder):
        os.makedirs(train_ant_xml_folder)
    if not os.path.exists(train_terrain_noise_folder):
        os.makedirs(train_terrain_noise_folder)

    folder_run_data: str = f"./runs/run_spec_{time.time()}"
    os.makedirs(folder_run_data, exist_ok=True)

    for env in tr_schedule.total_schedule[18:]:
        individuals: List[Individual] = [Individual(id=i) for i in range(parallel_jobs)]
        problem: AntProblem = AntProblem(individuals)
        searcher: XNES = XNES(problem, stdev_init=algo_stdev_init, popsize=24)
        stdout_logger: StdOutLogger = StdOutLogger(searcher)
        pandas_logger = PandasLogger(searcher)

        path_to_save: str = None
        if isinstance(env, RoughTerrain):
            path_to_save = f"{folder_run_data}/{type(env).__name__}_{env.block_size}_{env.floor_height}"
            os.makedirs(f"{path_to_save}/screenshots", exist_ok=True)
            os.makedirs(f"{path_to_save}/gen_tensors", exist_ok=True)
        elif isinstance(env, HillsTerrain):
            path_to_save = f"{folder_run_data}/{type(env).__name__}_{env.scale}_{env.floor_height}"
            os.makedirs(f"{path_to_save}/screenshots", exist_ok=True)
            os.makedirs(f"{path_to_save}/gen_tensors", exist_ok=True)
        elif isinstance(env, DefaultTerrain):
            path_to_save = f"{folder_run_data}/{type(env).__name__}"
            os.makedirs(f"{path_to_save}/screenshots", exist_ok=True)
            os.makedirs(f"{path_to_save}/gen_tensors", exist_ok=True)
        else:
            assert False, "Class type not supported"

        tr_schedule.training_schedule_partition = [env]
        best_generalist_ind: Tuple[torch.Tensor, float] = None
        num_generations_no_improvement: int = 0
        for GEN in range(spec_algo_max_generations + 1):
            searcher.step()
            for ind in individuals:
                ind.increment_generation()
            pop_best_params = searcher.status["pop_best"].values
            pop_best_fitness = searcher.status["pop_best_eval"]

            # Save screenshots
            if GEN % 10 == 0:
                individuals[0].setup_ant_default(pop_best_params)
                individuals[0].make_screenshot_ant(f"{path_to_save}/screenshots/ant_{GEN}.png")
                torch.save(pop_best_params, f"{path_to_save}/gen_tensors/{GEN}_population_best.pt")
            if GEN < spec_algo_init_training_generations:
                continue

            if best_generalist_ind == None or pop_best_fitness > best_generalist_ind[1]:
                best_generalist_ind = (pop_best_params, pop_best_fitness)
                torch.save(pop_best_params, f"{path_to_save}/gen_tensors/{GEN}_generalist_best_{pop_best_fitness}.pt")
                print(f"Current best fitness score: {pop_best_fitness}")
                num_generations_no_improvement = 0
            else:
                num_generations_no_improvement = num_generations_no_improvement + 1
            print(f"Number of generations ago when an improvement was found: {num_generations_no_improvement}")

            if num_generations_no_improvement >= spec_algo_gen_stagnation:
                break

        num_generations_no_improvement = 0
        G.append(best_generalist_ind[0])
        E.append([env])
        with open(f"{folder_run_data}/G_var.pkl", "wb") as file:
            pickle.dump(G, file)
        with open(f"{folder_run_data}/E_var.pkl", "wb") as file:
            pickle.dump(E, file)
        df = pandas_logger.to_dataframe()
        df.to_csv(f"{path_to_save}/gen_score_pandas_df.csv", index=False)
        create_plot(df, path_to_save)


def experiment1():
    """Runs experiment 1"""
    os.environ["MUJOCO_GL"] = "egl"
    algo: Experiment1 = Experiment1(23)
    algo.run()
    print("Experiment 1 has finished!!")


def experiment2():
    """Runs experiment 2"""
    os.environ["MUJOCO_GL"] = "egl"


def experiment3():
    """Runs experiment 3"""
    os.environ["MUJOCO_GL"] = "egl"


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
        "experiment1", help="Run Experiment 1 and generate a generalist for all environments."
    )

    parser_exp2 = subparsers.add_parser(
        "experiment2",
        help="Run Experiment 2 and generate a set of generalist that each handle a partition of the environments.",
    )

    parser_exp3 = subparsers.add_parser(
        "experiment3", help="Run Experiment 3 and generate a specialist for each environment."
    )

    parser_test = subparsers.add_parser(
        "test", help="Run a test experiment to visual a generated tensor from the tensor folders"
    )
    parser_test.add_argument("--tensor", type=Path, required=True, help="Path to the tensor that you want to test.")
    parser_test.add_argument(
        "--terrain", type=str, choices=TERRAIN_MAP.keys(), required=True, help="Terrain environment to test in"
    )
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
    elif args.experiment == "test":
        test_ant(args.tensor, args.terrain, args.params)
    else:
        print("Something went wrong!")


if __name__ == "__main__":
    main()
