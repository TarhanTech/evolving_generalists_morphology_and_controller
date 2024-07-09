from source.individual import Individual
from source.ant_problem import AntProblem
from source.globals import *
from source.training_env import *
from typing import List, Tuple
import time
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import pickle
import argparse
import os
import torch
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger
tr_schedule: TrainingSchedule = TrainingSchedule()

G = []
E = []

def partition(best_generalist_ind: Tuple[torch.Tensor, np.ndarray], individuals: List[Individual]):
    best_params = best_generalist_ind[0]

    all_fitness_scores_1 = validate_as_generalist(individuals, best_params)
    all_fitness_scores_2 = validate_as_generalist(individuals, best_params)
    all_fitness_scores_3 = validate_as_generalist(individuals, best_params)
    all_fitness_scores_mean = np.mean(np.array([all_fitness_scores_1, all_fitness_scores_2, all_fitness_scores_3]), axis=0)
    mean_fitness = np.mean(all_fitness_scores_mean)
    std_fitness = np.std(all_fitness_scores_mean)
    envs = []
    for i in range(len(tr_schedule.training_schedule) - 1, -1, -1):
        if all_fitness_scores_mean[i] > (mean_fitness - std_fitness): # fitness > mean - std
            envs.append(tr_schedule.remove_training_env(i))
    G.append(best_params)
    E.append(envs)

def validate(training_env, ind: Individual, params: torch.Tensor):
    if isinstance(training_env, RoughTerrain):
        ind.setup_ant_rough(params, training_env.floor_height, training_env.block_size)
    elif isinstance(training_env, HillsTerrain):
        ind.setup_ant_hills(params, training_env.floor_height, training_env.scale)
    elif isinstance(training_env, DefaultTerrain):
        ind.setup_ant_default(params)
    else:
        assert False, "Class type not supported"
    return ind.evaluate_fitness()

def validate_as_generalist(individuals: List[Individual], ind_best: torch.Tensor):
    batch_size: int = len(individuals)
    all_fitness = []
    
    for i in range(0, len(tr_schedule.training_schedule), batch_size):
        batch = tr_schedule.training_schedule[i:i + batch_size]
        tasks = (joblib.delayed(validate)(training_env, ind, ind_best) for training_env, ind in zip(batch, individuals))
        batch_fitness = joblib.Parallel(n_jobs=batch_size)(tasks)
        all_fitness.extend(batch_fitness)
    return np.array(all_fitness)

def test_ant(tensor_path: str):
    ind: Individual = Individual(id=99)
    params = torch.load(tensor_path)
    ind.setup_ant_hills(params, 2, 20)
    # ind.setup_ant_rough(params, 1, 4)
    # ind.setup_ant_default(params)
    total_reward: float = ind.evaluate_fitness(render_mode="human")
    print(f"Total Rewards: {total_reward}")

def train_ant():
    parallel_jobs: int = 6
    if not os.path.exists(train_ant_xml_folder):
        os.makedirs(train_ant_xml_folder)
    if not os.path.exists(train_terrain_noise_folder):
        os.makedirs(train_terrain_noise_folder)

    folder_run_data: str = f"./runs/run_{time.time()}"
    os.makedirs(folder_run_data, exist_ok=True)
    
    df_gen_scores = {"Generalist Score": []}
    partitions = 1
    try:
        while len(tr_schedule.training_schedule) != 0:
            individuals: List[Individual] = [Individual(id=i) for i in range(parallel_jobs)]
            problem : AntProblem = AntProblem(individuals)
            searcher: XNES = XNES(problem, stdev_init=algo_stdev_init, popsize=24)
            stdout_logger: StdOutLogger = StdOutLogger(searcher)

            os.makedirs(f"{folder_run_data}/partition_{partitions}/screenshots", exist_ok=True)
            os.makedirs(f"{folder_run_data}/partition_{partitions}/gen_tensors", exist_ok=True)
            print(f"The length of the training schedule is : {len(tr_schedule.training_schedule)}")

            best_generalist_ind: Tuple[torch.Tensor, float] = None
            num_generations_no_improvement: int = 0
            for GEN in range(algo_max_generations + 1):
                searcher.step()
                for ind in individuals: ind.increment_generation()
                pop_best_params = searcher.status["pop_best"].values
                individuals[0].setup_ant_default(pop_best_params)
                individuals[0].make_screenshot_ant(f"{folder_run_data}/partition_{partitions}/screenshots/ant_{GEN}.png")
                if GEN < algo_init_training_generations: continue

                gen_scores = validate_as_generalist(individuals, pop_best_params)
                mean_gen_score: float = np.mean(gen_scores)

                if best_generalist_ind == None or mean_gen_score > best_generalist_ind[1]:
                    best_generalist_ind = (pop_best_params, mean_gen_score)
                    torch.save(pop_best_params, f"{folder_run_data}/partition_{partitions}/gen_tensors/generalist_best_{GEN}.pt")
                    print(f"Current best generalist score: {mean_gen_score}")
                    num_generations_no_improvement = 0
                else:
                    num_generations_no_improvement = num_generations_no_improvement + 1
                print(f"Number of generations ago when an improvement was found: {num_generations_no_improvement}")
                df_gen_scores["Generalist Score"].append(mean_gen_score)
                pd.DataFrame(df_gen_scores).to_csv(f"{folder_run_data}/gen_score_pandas_df.csv", index=False)

                if num_generations_no_improvement >= algo_gen_stagnation:
                    num_generations_no_improvement = 0
                    partitions = partitions + 1
                    partition(best_generalist_ind, individuals)
                    with open(f"{folder_run_data}/G_var.pkl", "wb") as file:
                        pickle.dump(G, file)
                    with open(f"{folder_run_data}/E_var.pkl", "wb") as file:
                        pickle.dump(E, file)
                    break
        print("All environments are included in a partition! Algorithm ends.")
    except KeyboardInterrupt:
        with open(f"{folder_run_data}/G_var.pkl", "wb") as file:
            pickle.dump(G, file)
        with open(f"{folder_run_data}/E_var.pkl", "wb") as file:
            pickle.dump(E, file)


def main():
    parser = argparse.ArgumentParser(description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch")
    parser.add_argument("--tensor", type=str, help="Path to a tensor.pt file that should be tested")
    args = parser.parse_args()

    if args.tensor == None:
        train_ant()
    else:
        test_ant(args.tensor)

if __name__ == "__main__": main()