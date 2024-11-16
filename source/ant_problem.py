"""Definition of the custom optimization problem used for evotorch"""

from typing import List
from evotorch import Problem
import torch
import numpy as np
import evotorch
import joblib
from source.individual import Individual
from source.training_env import (
    TrainingSchedule,
    DefaultTerrain,
    HillsTerrain,
    RoughTerrain,
    TerrainType,
)


class AntProblem(Problem):
    """Definition of the custom optimization problem used for evotorch"""

    def __init__(
        self,
        device,
        tr_schedule: TrainingSchedule,
        individuals: List[Individual],
        initial_bounds: tuple[float, float],
        full_gen_algo: bool = False
    ):
        super().__init__(
            "max",
            solution_length=individuals[0].params_size,
            initial_bounds=initial_bounds,
            dtype=torch.float64,
            eval_dtype=torch.float64,
            device=device,
        )
        self.full_gen_algo: bool = full_gen_algo
        self.individuals: List[Individual] = individuals
        self.tr_schedule = tr_schedule

    def evals(self, params: torch.Tensor, ind: Individual) -> float:
        """Evaluate Individual on next training environment or on all environments if the full_gen_algo flag is true"""
        if self.full_gen_algo is True:
            fitnesses: list[float] = []
            for t_env in self.tr_schedule.training_terrains:
                ind.setup_env_ind(params, t_env)
                fitnesses.append(ind.evaluate_fitness())
            fitnesses = np.array(fitnesses)
            return np.mean(fitnesses)
        else:
            training_env: TerrainType = self.tr_schedule.get_training_terrain(
                ind.generation
            )
            ind.setup_env_ind(params, training_env)
            return ind.evaluate_fitness()

    def _evaluate_batch(self, batch: evotorch.SolutionBatch):
        batch_size: int = len(self.individuals)
        all_fitness = []

        # uncomment to easily debugs what happens in these threads
        # self.evals(batch.values[0], self.individuals[0])

        for i in range(0, len(batch.values), batch_size):
            batch_vals = batch.values[i : i + batch_size]
            tasks = (
                joblib.delayed(self.evals)(params, ind)
                for params, ind in zip(batch_vals, self.individuals)
            )
            batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
            all_fitness.extend(batch_fitness)
        batch.set_evals(torch.tensor(all_fitness, dtype=torch.float64))
