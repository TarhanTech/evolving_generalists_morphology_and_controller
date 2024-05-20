import evotorch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
from torch import Tensor

def sphere(x: Tensor) -> Tensor:
    return torch.sum(x.pow(2.0))

problem = Problem("min", sphere, solution_length=3, initial_bounds=(-1, 1))
searcher = SNES(problem, stdev_init=5)

stdout_logger: StdOutLogger = StdOutLogger(searcher)
pandas_logger: PandasLogger = PandasLogger(searcher)
searcher.run(100)

print(searcher.status["best"])

df = pandas_logger.to_dataframe()
df.to_csv("pandas_df.csv", index=False)
