from individual import *
from typing import List

from evotorch import Problem
import torch
import evotorch
import joblib

class AntProblem(Problem):
    def __init__(self, individuals: List[Individual]):
        nn_lower_bounds = [-0.00001] * individuals[0].controller.total_weigths
        morph_leg_length_lower_bounds = [individuals[0].morphology.leg_length_range[0]] * 8
        lower_bounds = nn_lower_bounds + morph_leg_length_lower_bounds

        nn_upper_bounds = [0.00001] * individuals[0].controller.total_weigths
        morph_leg_length_upper_bounds = [individuals[0].morphology.leg_length_range[0] + 0.1] * 8
        upper_bounds = nn_upper_bounds + morph_leg_length_upper_bounds
        
        super().__init__("max", solution_length=individuals[0].params_size, initial_bounds=(lower_bounds, upper_bounds), dtype=torch.float64, eval_dtype=torch.float64, device="cuda")
        self.individuals: List[Individual] = individuals
    
    def evals(self, params: torch.Tensor, ind: Individual) -> float:
        nn_params, morph_params = torch.split(params, (ind.controller.total_weigths, ind.morphology.total_params))
        ind.controller.set_nn_params(nn_params)
        ind.morphology.set_morph_params(morph_params)
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