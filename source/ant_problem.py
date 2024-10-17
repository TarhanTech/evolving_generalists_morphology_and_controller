"""Definition of the custom optimization problem used for evotorch"""

from typing import List
from evotorch import Problem
import torch
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

    def __init__(self, device, tr_schedule: TrainingSchedule, individuals: List[Individual], initial_bounds: tuple[float, float]):
        super().__init__(
            "max",
            solution_length=individuals[0].params_size,
            initial_bounds=initial_bounds,
            dtype=torch.float64,
            eval_dtype=torch.float64,
            device=device,
        )
        self.individuals: List[Individual] = individuals
        self.tr_schedule = tr_schedule

    def evals(self, params: torch.Tensor, ind: Individual) -> float:
        """Evaluate Individual on next training environment."""
        training_env: TerrainType = self.tr_schedule.get_training_terrain(ind.generation)

        if isinstance(training_env, RoughTerrain):
            ind.setup_ant_rough(params, training_env.floor_height, training_env.block_size)
        elif isinstance(training_env, HillsTerrain):
            ind.setup_ant_hills(params, training_env.floor_height, training_env.scale)
        elif isinstance(training_env, DefaultTerrain):
            ind.setup_ant_default(params)
        else:
            assert False, "Class type not supported"

        return ind.evaluate_fitness()

    def _evaluate_batch(self, batch: evotorch.SolutionBatch):
        batch_size: int = len(self.individuals)
        all_fitness = []

        # uncomment to easily debugs what happens in these threads
        # self.evals(batch.values[0], self.individuals[0])

        for i in range(0, len(batch.values), batch_size):
            batch_vals = batch.values[i : i + batch_size]
            tasks = (joblib.delayed(self.evals)(params, ind) for params, ind in zip(batch_vals, self.individuals))
            batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
            all_fitness.extend(batch_fitness)
        batch.set_evals(torch.tensor(all_fitness, dtype=torch.float64))
