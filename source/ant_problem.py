from source.individual import Individual
from source.mj_env import *
from typing import List
from source.globals import *

from evotorch import Problem
import torch
import evotorch
import joblib

class AntProblem(Problem):
    def __init__(self, individuals: List[Individual]):        
        super().__init__(
            "max", 
            solution_length=individuals[0].params_size, 
            initial_bounds=(algo_params_range[0], algo_params_range[1]), 
            dtype=torch.float64, 
            eval_dtype=torch.float64, 
            device="cuda"
        )
        self.individuals: List[Individual] = individuals
    
    def evals(self, params: torch.Tensor, ind: Individual) -> float:
        ind.setup_ant_rough(params, 0.1)
        return ind.evaluate_fitness()

    def _evaluate_batch(self, solutions: evotorch.SolutionBatch):
        batch_size: int = len(self.individuals)
        all_fitness = []
        
        for i in range(0, len(solutions.values), batch_size):
            batch = solutions.values[i:i + batch_size]
            tasks = (joblib.delayed(self.evals)(nn_params, nn) for nn_params, nn in zip(batch, self.individuals))
            batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
            all_fitness.extend(batch_fitness)
        solutions.set_evals(torch.tensor(all_fitness, dtype=torch.float64))