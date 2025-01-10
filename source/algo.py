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
from source.xnes_with_freeze import XNESWithFreeze
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

    def __init__(
        self,
        dis_morph_evo: bool,
        morph_type: str,
        use_custom_start_morph: bool,
        max_generations: int,
        gen_stagnation: int,
        init_training_generations: int,
        max_evals: int = None,
        parallel_jobs: int = 6,
        full_gen_algo: bool = False,
        freeze_params: str = None,
        freeze_interval: int = None,
    ):
        self.dis_morph_evo = dis_morph_evo
        self.morph_type = morph_type
        self.freeze_params: str = freeze_params
        self.freeze_interval: int = freeze_interval
        self.full_gen_algo: bool = full_gen_algo
        self.use_custom_start_morph: bool = use_custom_start_morph
        self.max_generations: int = max_generations
        self.gen_stagnation: int = gen_stagnation
        self.init_training_generations: int = init_training_generations

        self.t: TrainingSchedule = TrainingSchedule()
        self.g: List[Tensor] = []
        self.e: List[List[TerrainType]] = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parallel_jobs = parallel_jobs

        self.individuals: List[Individual] = self._initialize_individuals(self.dis_morph_evo, self.morph_type)
        self.searcher = None
        self.pandas_logger = None

        self.max_evals = max_evals
        if self.max_evals is not None:
            self.number_of_evals = 0

    @abstractmethod
    def run(self):
        """Run the experiment"""
        pass

    def _initialize_individuals(self, dis_morph_evo: bool, morph_type: str) -> List[Individual]:
        """Initialize the list of individuals."""

        return [
            Individual(
                self.device,
                self.morph_params_bounds_enc,
                self.penalty_growth_rate,
                self.penalty_scale_factor,
                self.penalty_scale_factor_err,
                dis_morph_evo,
                morph_type
            )
            for _ in range(self.parallel_jobs)
        ]

    def _initialize_searcher(self, old_searcher: XNES = None) -> XNES:
        """
        Initialize the XNES searcher. If an old searcher is provided,
        transfer its distribution into the newly created searcher.
        """
        problem = AntProblem(
            self.device,
            self.t,
            self.individuals,
            self.morph_params_bounds_enc,
            self.full_gen_algo,
            self.use_custom_start_morph
        )
        
        if self.freeze_params is None:
            self.searcher = XNES(problem, stdev_init=0.01)
        else:
            self.searcher = XNESWithFreeze(
                problem=problem,
                stdev_init=0.01,
                freeze_params=self.freeze_params,
                freeze_interval=self.freeze_interval
            )
        
        if old_searcher is not None:
            old_dist = old_searcher._distribution
            new_dim = problem.solution_length

            new_dist = self._slice_expgaussian_to_controller_only(old_dist, new_dim)

            print(f"old_searcher.step_count: {old_searcher.step_count}")
            self.searcher._steps_count = old_searcher.step_count
            self.searcher._distribution = new_dist

        self.pandas_logger = PandasLogger(self.searcher)
        StdOutLogger(self.searcher)
        
        return self.searcher

    def _slice_expgaussian_to_controller_only(self, old_dist, nn_params_size: int):
        """
        Given an ExpGaussian `old_dist` of dimension d,
        extract only the first `nn_params_size` coordinates
        (the 'controller' portion), building a new ExpGaussian
        of dimension `nn_params_size`.

        Args:
            old_dist: The existing ExpGaussian (as used by XNES),
                    dimension = d.
            nn_params_size: The dimension of the new distribution
                            (the 'controller' slice).

        Returns:
            A new ExpGaussian distribution of dimension `nn_params_size`
            with sub-blocks of mu, sigma, sigma_inv from old_dist.
        """

        import torch
        from evotorch.distributions import ExpGaussian

        old_mu = old_dist.mu
        new_mu = old_mu[:nn_params_size].clone()

        old_sigma = old_dist.sigma  # 2D
        new_sigma = old_sigma[:nn_params_size, :nn_params_size].clone()

        if "sigma_inv" in old_dist.parameters:
            old_sigma_inv = old_dist.sigma_inv
            new_sigma_inv = old_sigma_inv[:nn_params_size, :nn_params_size].clone()
        else:
            new_sigma_inv = None

        new_params = {
            "mu": new_mu,
            "sigma": new_sigma,
        }
        if new_sigma_inv is not None:
            new_params["sigma_inv"] = new_sigma_inv

        new_dist = ExpGaussian(
            parameters=new_params,
            solution_length=nn_params_size,
            device=old_dist.device,
            dtype=old_dist.dtype,
        )

        return new_dist

    def _set_individuals_generation(self, gen: int):
        for ind in self.individuals:
            ind.set_generation(gen)

    def _continue_search(self, num_generations_no_improvement: int, gen: int) -> bool:
        cond1: bool = num_generations_no_improvement < self.gen_stagnation
        cond2: bool = gen < self.max_generations

        cond3: bool = True
        if self.max_evals is not None:
            cond3 = self.number_of_evals < self.max_evals

        return cond1 and cond2 and cond3


class GeneralistExperimentBase(Algo):
    def __init__(
        self,
        dis_morph_evo: bool,
        morph_type: str,
        use_custom_start_morph,
        max_generations: int,
        gen_stagnation: int,
        init_training_generations: int,
        exp_folder_name: str,
        max_evals: int = None,
        parallel_jobs: int = 6,
        full_gen_algo: bool = False,
        freeze_params: str = None,
        freeze_interval: int = None,
        dis_morph_evo_later: bool = False,
    ):
        super().__init__(
            dis_morph_evo,
            morph_type,
            use_custom_start_morph,
            max_generations,
            gen_stagnation,
            init_training_generations,
            max_evals,
            parallel_jobs,
            full_gen_algo,
            freeze_params=freeze_params,
            freeze_interval=freeze_interval
        )

        self.dis_morph_evo_later: bool = dis_morph_evo_later
        self.ff_manager: FFManagerGeneralist = FFManagerGeneralist(exp_folder_name)

        self.df_gen_scores = {"Generalist Score": []}
        self.fitness_scores_dict = defaultdict(list)

    def run(self):
        """Run the experiment where you create a generalist for each partition of the environment."""
        partitions: int = 0
        while len(self.t.training_terrains) != 0 and (self.max_evals is not None and self.number_of_evals < self.max_evals):
            partitions += 1

            self.ff_manager.create_partition_folder(partitions)
            self.individuals = self._initialize_individuals(self.dis_morph_evo, self.morph_type)
            self._initialize_searcher()

            best_generalist, best_generalist_score = self._train(partitions)
            p_terrains: List[TerrainType] = self._partition(best_generalist)

            if self.dis_morph_evo_later:
                self.individuals = self._initialize_individuals(True, self.morph_type)
                for ind in self.individuals: 
                    _, morph_params = torch.split(
                        best_generalist, (self.individuals[0].controller.total_weigths, self.individuals[0].mj_env.morphology.total_params)
                    )
                    ind.mj_env.morphology.set_morph_params(morph_params)
                self._initialize_searcher(self.searcher)

            if self.max_evals is None or (self.max_evals is not None and self.number_of_evals < self.max_evals):
                self.t.setup_train_on_terrain_partition(p_terrains)
                best_generalist, _ = self._train(
                    partitions, best_generalist, best_generalist_score, self.dis_morph_evo_later
                )
                self.t.restore_training_terrains()

            if self.dis_morph_evo_later and best_generalist.numel() == self.individuals[0].controller.total_weigths:
                best_generalist = torch.cat([best_generalist, self.individuals[0].mj_env.morphology.morph_params_tensor])
            
            self.e.append(p_terrains)
            self.g.append(best_generalist)

            self._dump_logs(partitions)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)

    def _train(
        self,
        partitions,
        best_generalist: Tensor = None,
        best_generalist_score: float = float("-inf"),
        append_morph_to_tensor: bool = False,
    ) -> Tuple[Tensor, float]:
        num_generations_no_improvement: int = 0

        while (
            self._continue_search(
                num_generations_no_improvement, self.searcher.step_count
            )
            is True
        ):
            self.searcher.step()
            self._set_individuals_generation(self.searcher.step_count)
            pop_best: Tensor = self.searcher.status["pop_best"].values
            self.ff_manager.save_screenshot_ant(
                partitions, self.searcher.step_count, pop_best, self.individuals[0]
            )

            pop_best_to_save = pop_best
            if append_morph_to_tensor:
                pop_best_to_save = torch.cat([pop_best, self.individuals[0].mj_env.morphology.morph_params_tensor])


            fitness_scores: List[float] = self._validate_as_generalist(pop_best)
            generalist_score: float = np.mean(fitness_scores)
            if generalist_score > best_generalist_score:
                best_generalist = pop_best
                best_generalist_score = generalist_score
                self.ff_manager.save_generalist_tensor(
                    partitions, self.searcher.step_count, pop_best_to_save, True
                )
                num_generations_no_improvement = 0
            else:
                self.ff_manager.save_generalist_tensor(
                    partitions, self.searcher.step_count, pop_best_to_save, False
                )
                if self.init_training_generations < self.searcher.step_count:
                    num_generations_no_improvement += 1

            if self.max_evals is not None:
                self.number_of_evals += self.searcher._popsize + len(self.t.training_terrains)

            self.df_gen_scores["Generalist Score"].append(generalist_score)
            self.fitness_scores_dict[
                self.t.get_training_terrain(self.searcher.step_count).__str__()
            ].append(self.searcher.status["pop_best_eval"])
        

        self.ff_manager.log_evaluations(f"partition {partitions}", self.number_of_evals)
        return best_generalist, best_generalist_score

    def _validate_as_generalist(self, best_params: Tensor) -> np.ndarray[float]:
        all_fitness: List[float] = []

        for i in range(0, len(self.t.training_terrains), self.parallel_jobs):
            batch = self.t.training_terrains[i : i + self.parallel_jobs]
            tasks = (
                joblib.delayed(self._validate)(training_env, ind, best_params)
                for training_env, ind in zip(batch, self.individuals)
            )
            batch_fitness = joblib.Parallel(n_jobs=self.parallel_jobs)(tasks)
            all_fitness.extend(batch_fitness)
        return np.array(all_fitness)

    def _validate(
        self, training_env: TerrainType, ind: Individual, best_params: Tensor
    ) -> float:
        ind.setup_env_ind(best_params, training_env)
        return ind.evaluate_fitness()

    def _partition(self, best_params: Tensor) -> List[TerrainType]:
        count = 5
        all_fitness_scores: np.ndarray[float] = self._validate_as_generalist(
            best_params
        )
        for i in range(count - 1):
            fitness_scores: np.ndarray[float] = self._validate_as_generalist(
                best_params
            )
            all_fitness_scores = all_fitness_scores + fitness_scores

        all_fitness_scores_mean = all_fitness_scores / count
        mean_fitness = np.mean(all_fitness_scores_mean)
        std_fitness = np.std(all_fitness_scores_mean)
        print(f"Mean Fitness: {mean_fitness}")
        print(f"STD Fitness: {std_fitness}")
        envs: List[TerrainType] = []
        for i in range(len(self.t.training_terrains) - 1, -1, -1):
            if all_fitness_scores_mean[i] >= (
                mean_fitness - std_fitness
            ):  # fitness > mean - std
                envs.append(self.t.remove_training_terrain(i))
        return envs

    def _dump_logs(self, partitions: int):
        self.ff_manager.save_df(
            partitions, self.df_gen_scores, "gen_score_pandas_df.csv"
        )
        self.ff_manager.save_pandas_logger_df(partitions, self.pandas_logger)
        self.ff_manager.save_json(
            partitions, self.fitness_scores_dict, "fitness_scores.json"
        )

        self.df_gen_scores = {"Generalist Score": []}
        self.fitness_scores_dict = defaultdict(list)


class SpecialistExperimentBase(Algo):
    def __init__(
        self,
        dis_morph_evo: bool,
        max_generations: int,
        gen_stagnation: int,
        init_training_generations: int,
        exp_folder_name: str,
        parallel_jobs: int = 6,
    ):
        super().__init__(
            dis_morph_evo=dis_morph_evo,
            morph_type=None,
            max_generations=max_generations,
            gen_stagnation=gen_stagnation,
            init_training_generations=init_training_generations,
            parallel_jobs=parallel_jobs,
        )

        self.ff_manager: FFManagerSpecialist = FFManagerSpecialist(exp_folder_name)

    def _train(self, terrain: TerrainType) -> Tensor:
        num_generations_no_improvement: int = 0
        best_fitness: float = float("-inf")

        while (
            self._continue_search(
                num_generations_no_improvement, self.searcher.step_count
            )
            is True
        ):
            self.searcher.step()
            self._set_individuals_generation(self.searcher.step_count)

            pop_best: Tensor = self.searcher.status["pop_best"].values
            pop_best_fitness: float = self.searcher.status["pop_best_eval"]

            self.ff_manager.save_screenshot_ant(
                terrain, self.searcher.step_count, pop_best, self.individuals[0]
            )

            if pop_best_fitness > best_fitness:
                self.ff_manager.save_specialist_tensor(
                    terrain, self.searcher.step_count, pop_best, True
                )
                best_fitness = pop_best_fitness
                num_generations_no_improvement = 0
            else:
                self.ff_manager.save_specialist_tensor(
                    terrain, self.searcher.step_count, pop_best, False
                )
                if self.init_training_generations < self.searcher.step_count:
                    num_generations_no_improvement += 1

        return self.searcher.status["pop_best"].values


class OurAlgo(GeneralistExperimentBase):
    """
    Class used to run our algorithm where you create a generalist for each partition of the environments.

    1. Our algorithm with morphological evolution
    2. Our algorithm without morphological evolution (default morphology)
    3. Our algorithm without morphological evolution (large morphology)
    """

    def __init__(
        self, dis_morph_evo: bool, morph_type: bool, use_custom_start_morph: bool, dis_morph_evo_later: bool, parallel_jobs: int = 6, freeze_params: str = None, freeze_interval: int = None,
    ):
        if dis_morph_evo is False and morph_type is not None:
            raise Exception("Invalid argument combination: dis_morph_evo and specifying morph type is not possible. Morphology will be evolved")
        if use_custom_start_morph and (morph_type is not None or dis_morph_evo is True):
            raise Exception("Invalid argument combination: use_custom_start_morph can only be done with morphological evolution")
        if (freeze_params is None) != (freeze_interval is None):
            raise Exception("Invalid argument combination: freeze_params and freeze_interval must both be None or contain a value")

        exp_folder_name: str = ""
        if freeze_params == "morphology":
            exp_folder_name = f"OurAlgo-MorphEvo-FreezeMorph{freeze_interval}"
        elif freeze_params == "controller":
            exp_folder_name = f"OurAlgo-MorphEvo-FreezeContr{freeze_interval}"
        elif dis_morph_evo_later:
            exp_folder_name = "OurAlgo-MorphEvo-DisMorphLater-Gen"
        elif use_custom_start_morph:
            exp_folder_name = "OurAlgo-MorphEvo-StartLarge-Gen"
        elif dis_morph_evo is False:
            exp_folder_name = "OurAlgo-MorphEvo-Gen"
        elif dis_morph_evo and morph_type == "default":
            exp_folder_name = "OurAlgo-DefaultMorph-Gen"
        elif dis_morph_evo and morph_type == "large":
            exp_folder_name = "OurAlgo-LargeMorph-Gen"
        elif dis_morph_evo and morph_type == "custom":
            exp_folder_name = "OurAlgo-CustomMorph-Gen"
        else:
            raise ValueError(
                "Undefined experiment configuration: dis_morph_evo and morph_type combination is not handled."
            )

        super().__init__(
            dis_morph_evo=dis_morph_evo,
            morph_type=morph_type,
            use_custom_start_morph=use_custom_start_morph,
            max_generations=10000,
            gen_stagnation=200,
            init_training_generations=0,
            max_evals=150000,
            exp_folder_name=exp_folder_name,
            parallel_jobs=parallel_jobs,
            freeze_params=freeze_params,
            freeze_interval=freeze_interval,
            dis_morph_evo_later=dis_morph_evo_later,
        )


# Not used in paper
class OurAlgoOneGen(GeneralistExperimentBase):
    """Class used to run the experiment where you create one generalist for all the environments"""

    def __init__(self, parallel_jobs: int = 6):
        super().__init__(
            max_generations=10000,
            gen_stagnation=10000,
            init_training_generations=10000,
            exp_folder_name="OurAlgoNoPart-MorphEvo-Gen",
            parallel_jobs=parallel_jobs,
        )

    def run(self):
        """Run the experiment where you create one generalist for all the environments"""
        partitions: int = 1

        self.ff_manager.create_partition_folder(partitions)
        self._initialize_searcher()
        best_generalist, _ = self._train(partitions)

        self.e.append(self.t.training_terrains)
        self.g.append(best_generalist)

        self._dump_logs(partitions)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)

class FullGeneralist(GeneralistExperimentBase):
    """
    Class used to run an algorithm to create a full generalist. In here, as an objective function, 
    we validate the fitness on all training environments and take the average.

    1. Full generalist with morphological evolution.
    2. Full generalist without morphological evolution (default morphology).
    """

    """
    Our algorithm in these experiments lets the iterations/generations go till max 5000. We will assume this number for our calculation.
    The problem creates a population of 23 individuals. This means we have 23 evaluations per iterations.
    After every generation, we calculate the generalist score using the training environments (33 environments).
    In this calculation, I am assuming it partitions 3 times.

    |evals| = 5000(23 + 33) + 5000(23 + 7) + 5000(23 + 1) = 550.000
    |evals| = 280.000 + 150.000 + 120.000 = 550.000

    We want the same amount of evaluations for this algorithm. Full generalist takes:
    |evals| = num_generations * 23 * 33 = 550.000
    Solving for num_generations gives:
    num_generations = 725

    Partitioning in this algorithm is not possible in any way.
    """

    def __init__(
        self, dis_morph_evo: bool, morph_type: str, parallel_jobs: int = 6
    ):
        if dis_morph_evo is False and morph_type is not None:
            raise Exception("Invalid argument combination: dis_morph_evo and specifying morph type is not possible. Morphology will be evolved")
        
        exp_folder_name: str = ""
        if dis_morph_evo and morph_type == "default":
            exp_folder_name = "FullGen-DefaultMorph-Gen"
        elif dis_morph_evo is False:
            exp_folder_name = "FullGen-MorphEvo-Gen"
        else:
            raise ValueError(
                "Undefined experiment configuration"
            )

        super().__init__(
            dis_morph_evo=dis_morph_evo,
            morph_type="default",
            max_generations=725,
            gen_stagnation=725,
            init_training_generations=725,
            exp_folder_name=exp_folder_name,
            parallel_jobs=parallel_jobs,
            full_gen_algo=True
        )
    
    def run(self):
        """Run the experiment where you create one generalist for all the environments"""
        partitions: int = 1

        self.ff_manager.create_partition_folder(partitions)
        self._initialize_searcher()
        best_generalist, _ = self._train(partitions)

        self.e.append(self.t.training_terrains)
        self.g.append(best_generalist)

        self._dump_logs(partitions)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)


class Specialist(SpecialistExperimentBase):
    """
    Class used to run the specialist experiment where you create a specialist for each of the environments
    1. Specialists with morphological evolution.
    2. Specialists with morphological evolution (long generations).
    3. Specialists without morphological evolution (default morphology).
    """

    """
    Our algorithm in these experiments lets the iterations/generations go till max 5000. We will assume this number for our calculation.
    The problem creates a population of 23 individuals. This means we have 23 evaluations per iterations.
    After every generation, we calculate the generalist score using the training environments (33 environments).
    In this calculation, I am assuming it partitions 3 times.

    |evals| = 5000(23 + 33) + 5000(23 + 7) + 5000(23 + 1) = 550.000
    |evals| = 280.000 + 150.000 + 120.000 = 550.000

    We need 81 specialist MC-pairs.
    |evals|/specialist =  550.000/81 = 6790 evals/specialist
    |generation|/specialist = 6790/23 = 295 generations/specialist
    """

    def __init__(self, dis_morph_evo: bool, long: bool, parallel_jobs: int = 6):
        exp_folder_name: str = ""
        if dis_morph_evo is False and long is False:
            exp_folder_name = "Spec-MorphEvo"
        elif dis_morph_evo and long is False:
            exp_folder_name = "Spec-DefaultMorph"
        elif dis_morph_evo is False and long:
            exp_folder_name = "Spec-MorphEvo-Long"
        else:
            raise ValueError(
                "Undefined experiment configuration: dis_morph_evo and long combination is not handled."
            )

        super().__init__(
            dis_morph_evo=dis_morph_evo,
            max_generations=10000 if long else 295,
            gen_stagnation=750 if long else 295,
            init_training_generations=2500 if long else 295,
            exp_folder_name=exp_folder_name,
            parallel_jobs=parallel_jobs,
        )

    def run(self):
        """run the third experiment where you create a specialist for each of the environments"""
        self.t.all_terrains = [
            RoughTerrain(0.1, 4),
            RoughTerrain(0.2, 4),
            RoughTerrain(0.3, 4),
            RoughTerrain(0.4, 4),
            RoughTerrain(0.5, 4),
            RoughTerrain(0.6, 4),
            RoughTerrain(0.7, 4),
            RoughTerrain(0.8, 4),
            RoughTerrain(0.9, 4),
            RoughTerrain(1.0, 3),
            RoughTerrain(1.0, 4),
        ]
        for terrain in self.t.all_terrains:
            self.t.setup_train_on_terrain_partition([terrain])
            self.ff_manager.create_terrain_folder(terrain)
            self._initialize_searcher()

            best_specialist = self._train(terrain)

            self.e.append([terrain])
            self.g.append(best_specialist)

            self.ff_manager.save_pandas_logger_df(terrain, self.pandas_logger)

        self.ff_manager.save_pickle("G_var.pkl", self.g)
        self.ff_manager.save_pickle("E_var.pkl", self.e)
