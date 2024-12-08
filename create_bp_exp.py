import argparse
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from source.training_env import TerrainType
from scipy.stats import kruskal
import pandas as pd
import scikit_posthocs as sp

def calculate_mean_fitnesses(file_path: Path) -> float:
    if not file_path.exists():
        print(f"Warning: File '{file_path}' not found.")
        return None

    with open(file_path, 'rb') as file:
        data: List[Tuple[TerrainType, float]] = pickle.load(file)

    return sum(fitness for _, fitness in data) / len(data) if data else 0.0

def get_test_and_training_means(exp_path: Path):
    test_fitness_means = []
    train_fitness_means = []

    if not exp_path.exists():
        print(f"Error: The path '{exp_path}' does not exist.")
        return test_fitness_means, train_fitness_means

    if not exp_path.is_dir():
        print(f"Error: The path '{exp_path}' is not a directory.")
        return test_fitness_means, train_fitness_means

    for run_folder in exp_path.iterdir():
        if run_folder.is_dir():
            test_data_path = run_folder / "env_fitnesses_test.pkl"
            train_data_path = run_folder / "env_fitnesses_training.pkl"

            test_mean = calculate_mean_fitnesses(test_data_path)
            train_mean = calculate_mean_fitnesses(train_data_path)

            if test_mean is not None:
                test_fitness_means.append(test_mean)
            if train_mean is not None:
                train_fitness_means.append(train_mean)
        else:
            print(f"Skipping non-folder item: {run_folder}")

    return test_fitness_means, train_fitness_means

def create_boxplots(test_data: List[float], train_data: List[float], exp_name: str, output_path: Path):
    sns.set(style="whitegrid")
    labels = ["Test"] * len(test_data) + ["Train"] * len(train_data)
    fitness_values = test_data + train_data

    data = pd.DataFrame({'Fitness': fitness_values, 'Group': labels})

    plt.figure(figsize=(5, 6))
    palette = {"Test": "magenta", "Train": "teal"}

    boxplot = sns.boxplot(
        x='Group',
        y='Fitness',
        data=data,
        width=0.3,
        palette=palette,
        linewidth=1,
        fliersize=0,
        boxprops=dict(facecolor='none'),
        showmeans=False
    )

    sns.stripplot(
        x='Group',
        y='Fitness',
        data=data,
        palette=palette,
        size=5,
        jitter=True,
        dodge=False
    )

    boxplot.set_title(f"{exp_name}: Fitness Distribution of Testing and Training Environments", fontsize=14, fontweight="bold")
    boxplot.set_ylabel("Fitness", fontsize=12)
    boxplot.set_xlabel("Environment Set", fontsize=12)
    boxplot.tick_params(labelsize=10)
    plt.xlim(-0.5, 1.5)

    stat, p_value_kruskal = kruskal(test_data, train_data)

    dunn_results = sp.posthoc_dunn(data, val_col='Fitness', group_col='Group', p_adjust='bonferroni')

    p_value_dunn = dunn_results.loc['Test', 'Train']

    def significance_markers(p):
        if p <= 0.001:
            return '***'
        elif p <= 0.01:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    sig_marker = significance_markers(p_value_dunn)

    y_max = max(fitness_values)
    y_min = min(fitness_values)
    y, h, col = y_max + (y_max - y_min) * 0.05, (y_max - y_min) * 0.02, 'k'
    x1, x2 = 0, 1

    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, sig_marker, ha='center', va='bottom', color=col)

    plt.savefig(output_path / "bp_fitness_test_training.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Iterate over folders in a specified directory.")
    parser.add_argument(
        "--exp_path",
        type=Path,
        help="The path to the directory to iterate over."
    )
    args = parser.parse_args()

    test_fitness_means, train_fitness_means = get_test_and_training_means(args.exp_path)

    create_boxplots(test_fitness_means, train_fitness_means, args.exp_path.parts[-1], Path())

if __name__ == "__main__":
    main()
