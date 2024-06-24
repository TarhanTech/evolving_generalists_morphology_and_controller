from source.individual import Individual
from source.mj_env import Morphology
from source.ant_problem import AntProblem
from source.globals import *
from typing import List
import time
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os
import torch
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger

def create_plot(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df["mean_eval"], label="Mean Evaluation", marker="o")
    plt.plot(df.index, df["pop_best_eval"], label="Population Best Evaluation", marker="s")
    plt.plot(df.index, df["median_eval"], label="Median Evaluation", marker="d")

    plt.xlabel("Generation")
    plt.ylabel("Evaluation Values")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig(f"{save_path}/evaluation_metrics_plot.png", dpi=300, bbox_inches="tight")

def train_ant():
    start_time = time.time()

    parallel_jobs: int = 12
    individuals: List[Individual] = [Individual(id=i) for i in range(parallel_jobs)]
    individuals[0].print_controller_info()

    problem : AntProblem = AntProblem(individuals)
    searcher: XNES = XNES(problem, stdev_init=algo_stdev_init, popsize=24)
    print(f"Pop size: {searcher._popsize}")

    stdout_logger: StdOutLogger = StdOutLogger(searcher)
    pandas_logger: PandasLogger = PandasLogger(searcher)
     
    if not os.path.exists(train_ant_xml_folder):
        os.makedirs(train_ant_xml_folder)
    if not os.path.exists(train_terrain_noise_folder):
        os.makedirs(train_terrain_noise_folder)

    folder_run_data: str = f"./runs/run_{start_time}"
    os.makedirs(folder_run_data, exist_ok=True)
    os.makedirs(f"{folder_run_data}/tensors", exist_ok=True)
    os.makedirs(f"{folder_run_data}/tensors_csv", exist_ok=True)
    os.makedirs(f"{folder_run_data}/screenshots", exist_ok=True)

    max_generations: int = 1500
    save_generation_rate: int = 50
    for i in range(save_generation_rate, max_generations + 1, save_generation_rate):
        for _ in range(save_generation_rate):
            searcher.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        df = pandas_logger.to_dataframe()
        df.to_csv(f"{folder_run_data}/evo_pandas_df.csv", index=False)
        create_plot(df, folder_run_data)

        pop_best_params: torch.Tensor = searcher.status["pop_best"].values
        individuals[0].setup_ant_default(pop_best_params)
        individuals[0].make_screenshot(f"{folder_run_data}/screenshots/ant_{i}.png")
        torch.save(pop_best_params, f"{folder_run_data}/tensors/pop_best_{i}.pt")

        pop_best_params_np = pop_best_params.to("cpu").detach().numpy()
        pop_best_params_df = pd.DataFrame(pop_best_params_np)
        pop_best_params_df.to_csv(f"{folder_run_data}/tensors_csv/pop_best_{i}.csv", index=False)

def test_ant(tensor_path: str):
    ind: Individual = Individual(id=99)
    params = torch.load(tensor_path)
    ind.setup_ant_rough(params, 0.2)
    total_reward: float = ind.evaluate_fitness(render_mode="human")
    print(f"Total Rewards: {total_reward}")

def main():
    parser = argparse.ArgumentParser(description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch")
    parser.add_argument("--tensor", type=str, help="Path to a tensor.pt file that should be tested")
    args = parser.parse_args()

    if args.tensor == None:
        train_ant()
    else:
        test_ant(args.tensor)

    

if __name__ == "__main__": main()