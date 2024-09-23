"""This module defines the parent class used to initiate one of the experimental runs."""

from collections import defaultdict
from typing import List, Tuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import numpy as np
import joblib
from evotorch.algorithms import XNES
from evotorch.logging import PandasLogger, StdOutLogger
from source.individual import Individual
from source.ant_problem import AntProblem
from source.ff_manager import FFManagerGeneralist, FFManagerSpecialist
from source.training_env import (
    TrainingSchedule,
    DefaultTerrain,
    RoughTerrain,
    HillsTerrain,
    TerrainType,
)


class Algo(ABC):
    """Parent class used as an interface for creating an experimental run"""

    morph_params_bounds_enc: tuple[float, float] = (-0.1, 0.1)
    penalty_growth_rate: float = 1.03
    penalty_scale_factor: int = 100
    penalty_scale_factor_err: int = 1000

    def __init__(self, parallel_jobs: int = 6):
        self.t: TrainingSchedule = TrainingSchedule()
        self.g: List[Tensor] = []
        self.e: List[List[TerrainType]] = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parallel_jobs = parallel_jobs

        self.individuals: List[Individual] = self._initialize_individuals()
        self.searcher = None
        self.pandas_logger = None

    @abstractmethod
    def run(self):
        """Run the experiment"""
        pass

    def _initialize_individuals(self) -> List[Individual]:
        """Initialize the list of individuals."""

        return [
            Individual(
                self.device,
                self.morph_params_bounds_enc,
                self.penalty_growth_rate,
                self.penalty_scale_factor,
                self.penalty_scale_factor_err,
            )
            for i in range(self.parallel_jobs)
        ]

    def _initialize_searcher(self) -> XNES:
        """Initialize the XNES searcher."""
        problem = AntProblem(self.device, self.t, self.individuals, self.morph_params_bounds_enc)
        self.searcher = XNES(problem, stdev_init=0.01)
        self.pandas_logger = PandasLogger(self.searcher)
        StdOutLogger(self.searcher)

    def _set_individuals_generation(self, gen: int):
        for ind in self.individuals:
            ind.set_generation(gen)

    def _continue_search(self, num_generations_no_improvement: int, gen: int) -> bool:
        cond1: bool = num_generations_no_improvement < self.gen_stagnation
        cond2: bool = gen < self.max_generations
        return cond1 and cond2


class Experiment1(Algo):
    """Class used to run the first experiment where you create a generalist for each partition of the environments."""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(parallel_jobs)

        self.init_training_generations: int = 2500
        self.max_generations: int = 5000
        self.gen_stagnation: int = 500

        self.ff_manager: FFManagerGeneralist = FFManagerGeneralist("exp1_gen")

        self.df_gen_scores = {"Generalist Score": []}
        self.df_fitness_scores = defaultdict(list)

    def run(self):
        """Run the experiment where you create a generalist for each partition of the environment."""
        partitions: int = 0
        while len(self.t.training_terrains) != 0:
            partitions += 1
            self.df_gen_scores = {"Generalist Score": []}
            self.df_fitness_scores = defaultdict(list)

            self.ff_manager.create_partition_folder(partitions)
            self._initialize_searcher()

            best_generalist, best_generalist_score = self._train(partitions)
            p_terrains: List[TerrainType] = self._partition(best_generalist)

            self.t.setup_train_on_terrain_partition(p_terrains)
            best_generalist, _ = self._train(partitions, best_generalist, best_generalist_score)
            self.t.restore_training_terrains()

            self.e.append(p_terrains)
            self.g.append(best_generalist)
            
            self.ff_manager.save_df(partitions, self.df_gen_scores, "gen_score_pandas_df.csv")
            self.ff_manager.save_df(partitions, self.df_fitness_scores, "fitness_scores_df.csv")
            self.ff_manager.save_pandas_logger_df(partitions, self.pandas_logger)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)

    def _train(self, partitions, best_generalist: Tensor = None, best_generalist_score: float = float("-inf")) -> Tuple[Tensor, float]:
        num_generations_no_improvement: int = 0

        while self._continue_search(num_generations_no_improvement, self.searcher.step_count) is True:
            self.searcher.step()
            self._set_individuals_generation(self.searcher.step_count)
            pop_best: Tensor = self.searcher.status["pop_best"].values

            self.ff_manager.save_screenshot_ant(partitions, self.searcher.step_count, pop_best, self.individuals[0])

            fitness_scores: List[float] = self._validate_as_generalist(pop_best)
            generalist_score: float = np.mean(fitness_scores)
            if generalist_score > best_generalist_score:
                best_generalist = pop_best
                best_generalist_score = generalist_score
                self.ff_manager.save_generalist_tensor(partitions, self.searcher.step_count, pop_best, True)
                num_generations_no_improvement = 0
            else:
                self.ff_manager.save_generalist_tensor(partitions, self.searcher.step_count, pop_best, False)
                if self.init_training_generations < self.searcher.step_count:
                    num_generations_no_improvement += 1

            self.df_gen_scores["Generalist Score"].append(generalist_score)
            self.df_fitness_scores[self.t.get_current_training_terrain().__str__()].append(self.searcher.status["pop_best_eval"])
        return best_generalist, best_generalist_score

    def _validate_as_generalist(self, best_params: Tensor) -> np.ndarray[float]:
        all_fitness: List[float] = []

        for i in range(0, len(self.t.training_terrains), self.parallel_jobs):
            batch = self.t.training_terrains[i : i + self.parallel_jobs]
            tasks = (joblib.delayed(self._validate)(training_env, ind, best_params) for training_env, ind in zip(batch, self.individuals))
            batch_fitness = joblib.Parallel(n_jobs=self.parallel_jobs)(tasks)
            all_fitness.extend(batch_fitness)
        return np.array(all_fitness)

    def _validate(self, training_env: TerrainType, ind: Individual, best_params: Tensor) -> float:
        if isinstance(training_env, RoughTerrain):
            ind.setup_ant_rough(best_params, training_env.floor_height, training_env.block_size)
        elif isinstance(training_env, HillsTerrain):
            ind.setup_ant_hills(best_params, training_env.floor_height, training_env.scale)
        elif isinstance(training_env, DefaultTerrain):
            ind.setup_ant_default(best_params)
        else:
            assert False, "Class type not supported"
        return ind.evaluate_fitness()

    def _partition(self, best_params: Tensor) -> List[TerrainType]:
        count = 5
        all_fitness_scores: np.ndarray[float] = self._validate_as_generalist(best_params)
        for i in range(count - 1):
            fitness_scores: np.ndarray[float] = self._validate_as_generalist(best_params)
            all_fitness_scores = all_fitness_scores + fitness_scores

        all_fitness_scores_mean = all_fitness_scores / count
        mean_fitness = np.mean(all_fitness_scores_mean)
        std_fitness = np.std(all_fitness_scores_mean)
        print(f"Mean Fitness: {mean_fitness}")
        print(f"STD Fitness: {std_fitness}")
        envs: List[TerrainType] = []
        for i in range(len(self.t.training_terrains) - 1, -1, -1):
            if all_fitness_scores_mean[i] >= (mean_fitness - std_fitness):  # fitness > mean - std
                envs.append(self.t.remove_training_terrain(i))
        return envs


class Experiment2(Algo):
    """Class used to run the second experiment where you create one generalist for all the environments"""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(parallel_jobs)

        self.init_training_generations: int = 10000
        self.max_generations: int = 10000
        self.gen_stagnation: int = 10000

        self.ff_manager: FFManagerGeneralist = FFManagerGeneralist("exp2_gen")

        self.df_gen_scores = {"Generalist Score": []}

    def run(self):
        """Run the experiment where you create one generalist for all the environments"""
        pass


class Experiment3(Algo):
    """Class used to run the third experiment where you create a specialist for each of the environments"""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(parallel_jobs)

        self.init_training_generations: int = 2500
        self.max_generations: int = 10000
        self.gen_stagnation: int = 750

        self.ff_manager: FFManagerSpecialist = FFManagerSpecialist("exp3_spec")

    def run(self):
        """run the third experiment where you create a specialist for each of the environments"""
        terrains_to_create_specialist = self.t.all_terrains[0::10]
        for terrain in terrains_to_create_specialist:
            self.t.setup_train_on_terrain_partition([terrain])
            self.ff_manager.create_terrain_folder(terrain)
            self._initialize_searcher()

            best_specialist = self._train(terrain)

            self.e.append([terrain])
            self.g.append(best_specialist)

            self.ff_manager.save_pandas_logger_df(terrain, self.pandas_logger)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)

    def _train(self, terrain: TerrainType) -> Tensor:
        num_generations_no_improvement: int = 0
        best_fitness: float = float("-inf")

        while self._continue_search(num_generations_no_improvement, self.searcher.step_count) is True:
            self.searcher.step()
            self._set_individuals_generation(self.searcher.step_count)

            pop_best: Tensor = self.searcher.status["pop_best"].values
            pop_best_fitness: float = self.searcher.status["pop_best_eval"]

            self.ff_manager.save_screenshot_ant(terrain, self.searcher.step_count, pop_best, self.individuals[0])

            if pop_best_fitness > best_fitness:
                best_fitness = pop_best_fitness
                num_generations_no_improvement = 0
            else:
                if self.init_training_generations < self.searcher.step_count:
                    num_generations_no_improvement += 1
            self.ff_manager.save_specialist_tensor(terrain, self.searcher.step_count, pop_best)

        return self.searcher.status["pop_best"].values


class Experiment4(Algo):
    """Class used to run the fourth experiment where you create a specialist for each of the environments using same resources as experiment 1"""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(parallel_jobs)

        self.max_generations: int = 10000

        self.ff_manager: FFManagerGeneralist = FFManagerGeneralist("exp4_spec")


class Experiment5(Algo):
    """Class used to run the fifth experiment where you create a generalist (no morphological evolution) for each partition of the environments."""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(parallel_jobs)

        self.ff_manager: FFManagerGeneralist = FFManagerGeneralist("exp5_gen")

    def run(self):
        pass
