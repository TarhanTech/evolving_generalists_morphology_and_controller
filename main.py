from individual import *
from ant_problem import *
from typing import List
import time
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import evotorch
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

def test_ant(tensor_path: str):
    ind: Individual = Individual()
    tensor = torch.load(tensor_path)
    nn_params, morph_params = torch.split(tensor, (ind.controller.total_weigths, ind.morphology.total_params))
    print(nn_params)
    print(morph_params)
    ind.controller.set_nn_params(nn_params)
    ind.morphology.set_morph_params(morph_params)
    ind.evaluate_fitness_rendered()

def train_ant():
    start_time = time.time()
    parallel_jobs: int = 12

    individuals: List[Individual] = [Individual() for _ in range(parallel_jobs)]
    individuals[0].print_morph_info()
    individuals[0].print_controller_info()

    problem : AntProblem = AntProblem(individuals)
    searcher: XNES = XNES(problem, stdev_init=0.01, popsize=24)
    print(f"Pop size: {searcher._popsize}")

    stdout_logger: StdOutLogger = StdOutLogger(searcher)
    pandas_logger: PandasLogger = PandasLogger(searcher)
    
    folder_run_data: str = f"./runs/run_{start_time}"
    os.makedirs(folder_run_data, exist_ok=True)
    os.makedirs(f"{folder_run_data}/tensors", exist_ok=True)
    os.makedirs(f"{folder_run_data}/tensors_csv", exist_ok=True)

    max_generations: int = 100
    save_generation_rate: int = 20
    for i in range(save_generation_rate, max_generations + 1, save_generation_rate):
        for _ in range(save_generation_rate):
            searcher.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        df = pandas_logger.to_dataframe()
        df.to_csv(f"{folder_run_data}/evo_pandas_df.csv", index=False)
        create_plot(df, folder_run_data)

        pop_best_solution: torch.Tensor = searcher.status["pop_best"].values
        torch.save(pop_best_solution, f"{folder_run_data}/tensors/pop_best_{i}.pt")

        pop_best_solution_np = pop_best_solution.to("cpu").detach().numpy()
        pop_best_solution_df = pd.DataFrame(pop_best_solution_np)
        pop_best_solution_df.to_csv(f"{folder_run_data}/tensors_csv/pop_best_{i}.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch")
    parser.add_argument("--tensor", type=str, help="Path to a tensor.pt file that should be tested")
    args = parser.parse_args()

    if args.tensor == None:
        train_ant()
    else:
        test_ant(args.tensor)

    

if __name__ == "__main__": main()