import pickle
import numpy as np
from pathlib import Path
from scipy import stats

def calculate_per_run_std(run_folder: Path) -> float:
    """
    Loads all fitnesses from env_fitnesses_test.pkl and env_fitnesses_training.pkl.
    Returns the std of those fitness scores for this run folder.
    """
    test_data_path = run_folder / "env_fitnesses_test.pkl"
    train_data_path = run_folder / "env_fitnesses_training.pkl"

    data = []
    if test_data_path.exists():
        with open(test_data_path, 'rb') as f:
            data.extend(pickle.load(f))
    if train_data_path.exists():
        with open(train_data_path, 'rb') as f:
            data.extend(pickle.load(f))

    # Extract the fitness values
    fitness_values = [fitness for _, fitness in data]

    # Return the sample standard deviation (ddof=1) if we have any fitness data
    return float(np.std(fitness_values, ddof=1)) if len(fitness_values) > 1 else 0.0

def gather_std_for_experiment(experiment_path: Path) -> list[float]:
    """
    Iterates over each run folder in the experiment path,
    calculates the per-run std of fitnesses, and returns a list of those stds.
    """
    run_stds = []
    for run_folder in experiment_path.iterdir():
        if run_folder.is_dir():  # Only consider directories
            run_std = calculate_per_run_std(run_folder)
            run_stds.append(run_std)
    return run_stds

def compare_experiments(exp_path1: Path, exp_path2: Path):
    """
    Gathers the per-run-folder std of fitnesses for each experiment,
    then performs a two-sample t-test (Welch’s) to check if
    the mean of those stds differs significantly between experiments.
    """
    stds1 = gather_std_for_experiment(exp_path1)
    stds2 = gather_std_for_experiment(exp_path2)

    mean1, mean2 = np.mean(stds1), np.mean(stds2)
    std1, std2 = np.std(stds1, ddof=1), np.std(stds2, ddof=1)

    print(f"--- Experiment 1: {exp_path1} ---")
    print(f"  Number of runs: {len(stds1)}")
    print(f"  Mean of per-run stds: {mean1:.3f}")
    print(f"  Std of per-run stds:  {std1:.3f}\n")

    print(f"--- Experiment 2: {exp_path2} ---")
    print(f"  Number of runs: {len(stds2)}")
    print(f"  Mean of per-run stds: {mean2:.3f}")
    print(f"  Std of per-run stds:  {std2:.3f}\n")

    # Welch’s t-test (does not assume equal variance)
    t_stat, p_val = stats.ttest_ind(stds1, stds2, equal_var=False)

    print("Two-sample t-test (Welch’s) on the means of per-run stds:")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4g}")

    alpha = 0.05
    if p_val < alpha:
        print(f" Statistically significant at alpha={alpha}\n")
    else:
        print(f"Not statistically significant at alpha={alpha}.\n")

def main():
    # Adjust these paths to point to your experiment directories
    exp_path1 = Path("./runs/OurAlgo-MorphEvo-Gen")
    exp_path2 = Path("./runs/OurAlgo-LargeMorph-Gen")

    compare_experiments(exp_path1, exp_path2)

if __name__ == "__main__":
    main()
