from individual import *
from ant_problem import *
from typing import List
import time
import pandas as pd
import matplotlib.pyplot as plt

import evotorch
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger

def create_plot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df['mean_eval'], label='Mean Evaluation', marker='o')
    plt.plot(df.index, df['pop_best_eval'], label='Population Best Evaluation', marker='s')
    plt.plot(df.index, df['median_eval'], label='Median Evaluation', marker='d')

    plt.xlabel('Generation')
    plt.ylabel('Evaluation Values')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig('evaluation_metrics_plot.png', dpi=300, bbox_inches='tight')


def main():
    start_time = time.time()
    parallel_jobs: int = 6

    individuals: List[Individual] = [Individual() for _ in range(parallel_jobs)]
    individuals[0].print_morph_info()
    individuals[0].print_controller_info()

    problem : AntProblem = AntProblem(individuals)
    searcher: XNES = XNES(problem, stdev_init=0.01, popsize=24, )
    print(f"Pop size: {searcher._popsize}")

    stdout_logger: StdOutLogger = StdOutLogger(searcher)
    pandas_logger: PandasLogger = PandasLogger(searcher)
    
    
    for i in range(0, 10):
        for _ in range(100):
            searcher.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        df = pandas_logger.to_dataframe()
        df.to_csv("evo_pandas_df.csv", index=False)
        create_plot(df)

        pop_best_solution: torch.Tensor = searcher.status["pop_best"].values
        torch.save(pop_best_solution, f"./tensors/pop_best_{i}.pt")

        pop_best_solution_np = pop_best_solution.to("cpu").detach().numpy()
        pop_best_solution_df = pd.DataFrame(pop_best_solution_np)
        # np.savetxt(f"./tensors_csv/pop_best_{i}.csv", pop_best_solution_np, delimiter=',', fmt='%d')
        pop_best_solution_df.to_csv(f"./tensors_csv/pop_best_{i}.csv", index=False)


if __name__ == "__main__": main()