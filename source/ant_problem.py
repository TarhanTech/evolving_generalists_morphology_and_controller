from typing import List
import numpy as np
import torch
import joblib
import evotorch
from evotorch import Problem, SolutionBatch
from source.individual import Individual
from source.training_env import TrainingSchedule, TerrainType
from source.mj_env import Morphology


class AntProblem(Problem):
    """Definition of the custom optimization problem used for evotorch"""

    def __init__(
        self,
        device,
        tr_schedule: TrainingSchedule,
        individuals: List[Individual],
        initial_bounds: tuple[float, float],  # e.g. (-1.0, 1.0)
        full_gen_algo: bool = False,
        use_custom_start_morph: bool = False
    ):
        param_bounds = initial_bounds
        self.nn_params_size = individuals[0].controller.total_weigths
        self.morph_params_size = individuals[0].mj_env.morphology.total_params
        if use_custom_start_morph:

            # For convenience, break out the scalar bounds:
            low, high = initial_bounds   # e.g. low=-1.0, high=1.0

            # Build per-parameter bounds array, shape (2, solution_length)
            param_bounds = np.zeros((2, solution_length), dtype=np.float64)

            # 1) For the NN portion, allow [-1.0, 1.0] (example)
            param_bounds[0, :self.nn_params_size] = low
            param_bounds[1, :self.nn_params_size] = high

            # 2) For the morphological portion, pin them to a single value
            #    We'll replicate your idea of default_values:
            #    [high] * total_leg_length_params + [0.08] * total_leg_width_params
            morph_default_values = np.array(
                [high] * Morphology.total_leg_length_params  # e.g. set leg lengths to `high`
                + [-0.06] * Morphology.total_leg_width_params  # set widths to 0.08
            )

            morph_start = self.nn_params_size
            morph_end = morph_start + self.morph_params_size

            # Pin the morphological portion by making the low bound == high bound == that default value
            param_bounds[0, morph_start:morph_end] = morph_default_values
            param_bounds[1, morph_start:morph_end] = morph_default_values

        super().__init__(
            objective_sense="max",
            solution_length=individuals[0].params_size,
            initial_bounds=param_bounds,      
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
            return float(np.mean(fitnesses))
        else:
            training_env: TerrainType = self.tr_schedule.get_training_terrain(ind.generation)
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
