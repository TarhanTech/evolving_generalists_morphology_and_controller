import os
import pickle
from pathlib import Path
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal, ranksums

# Hardcoded base runs path
BASE_RUNS_PATH = Path("./runs/_olds_or_wrong/GenStagnation300_MaxEvals303600")

EXPERIMENT_PATHS = {
    "FullGen-DefaultMorph-Gen": BASE_RUNS_PATH / "FullGen-DefaultMorph-Gen",
    "FullGen-MorphEvo-Gen": BASE_RUNS_PATH / "FullGen-MorphEvo-Gen",
    # "OurAlgo-CustomMorph-Gen": BASE_RUNS_PATH / "OurAlgo-CustomMorph-Gen",
    "OurAlgo-DefaultMorph-Gen": BASE_RUNS_PATH / "OurAlgo-DefaultMorph-Gen",
    "OurAlgo-LargeMorph-Gen": BASE_RUNS_PATH / "OurAlgo-LargeMorph-Gen",
    "OurAlgo-MorphEvo-Gen": BASE_RUNS_PATH / "OurAlgo-MorphEvo-Gen",
    "OurAlgo-MorphEvo-StartLarge-Gen": BASE_RUNS_PATH / "OurAlgo-MorphEvo-StartLarge-Gen", 
    "Spec-DefaultMorph": BASE_RUNS_PATH / "Spec-DefaultMorph",
    "Spec-MorphEvo": BASE_RUNS_PATH / "Spec-MorphEvo",
    "Spec-MorphEvo-Long": BASE_RUNS_PATH / "Spec-MorphEvo-Long"
}

def get_last_evaluations(run_folder):
    evals_file = run_folder / "number_of_evals.log"
    if evals_file.exists():
        with open(evals_file, "r") as f:
            lines = f.readlines()
            if lines:
                # Extract the last line with the number of evaluations
                last_line = lines[-1]
                try:
                    # Extract the last number in the line
                    last_number = int(last_line.split()[-1])
                    return last_number
                except (IndexError, ValueError):
                    pass
    return "unknown"

def calculate_mean_fitness(run_folder: Path) -> float:
    test_data_path = run_folder / "env_fitnesses_test.pkl"
    train_data_path = run_folder / "env_fitnesses_training.pkl"

    data = []
    if test_data_path.exists():
        with open(test_data_path, 'rb') as file:
            data.extend(pickle.load(file))
    if train_data_path.exists():
        with open(train_data_path, 'rb') as file:
            data.extend(pickle.load(file))

    return sum(fitness for _, fitness in data) / len(data) if data else 0.0


def get_experiment_fitness_means(exp_path: Path) -> List[float]:
    fitness_means = []
    evaluations = []

    if not exp_path.exists():
        print(f"Error: The path '{exp_path}' does not exist.")
        return fitness_means

    if not exp_path.is_dir():
        print(f"Error: The path '{exp_path}' is not a directory.")
        return fitness_means

    for run_folder in exp_path.iterdir():
        if run_folder.is_dir():
            mean = calculate_mean_fitness(run_folder)
            fitness_means.append(mean)

            evals = get_last_evaluations(run_folder)
            evaluations.append(evals)

    df = pd.DataFrame({
        "Fitness": fitness_means,
        "Evaluations": evaluations
    })

    # Save each experiment's fitness data to a CSV (optional)
    folder_path = "fitness_df"
    file_name = exp_path.name + ".csv"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)

    return fitness_means


def significance_markers(p_value):
    """Returns a star marker based on p-value thresholds."""
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'ns'


def create_multigroup_boxplot(data: pd.DataFrame, output_path: Path):
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    groups = data['Group'].unique()
    order = list(groups)

    palette = sns.color_palette("tab10", n_colors=len(groups))
    palette = dict(zip(groups, palette))

    # === Boxplot with Whiskers and Outliers ===
    boxplot = sns.boxplot(
        x='Group',
        y='Fitness',
        data=data,
        order=order,
        palette=palette,
        width=0.5,
        linewidth=1,
        fliersize=0,
        boxprops=dict(facecolor='none'),
        showmeans=False
    )

    # Add jittered points for each data sample (optional)
    sns.stripplot(
        x='Group',
        y='Fitness',
        data=data,
        order=groups,
        palette=palette,
        size=5,
        jitter=True,
        dodge=False,
        alpha=0.7
    )

    boxplot.set_title("Comparison of All Experiments", fontsize=14, fontweight="bold")
    boxplot.set_ylabel("Fitness", fontsize=12)
    boxplot.set_xlabel("Experiments", fontsize=12)
    boxplot.tick_params(labelsize=10)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # -----------------------
    # 1. Kruskal-Wallis Test
    # -----------------------
    group_data = [data.loc[data['Group'] == g, 'Fitness'] for g in groups]
    stat, p_value_kruskal = kruskal(*group_data)
    print(f"Kruskal-Wallis test: H={stat:.3f}, p={p_value_kruskal:.3e}")
    if p_value_kruskal < 0.05:
        print("=> At least one group differs significantly (Kruskal–Wallis).")
    else:
        print("=> No significant difference among all groups (Kruskal–Wallis).")

    # ------------------------------------------------------
    # 2. Pairwise Wilcoxon Rank-Sum (NO multiple correction)
    # ------------------------------------------------------
    print("\nPairwise Wilcoxon rank-sum tests (no correction):")
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            arr1 = data.loc[data['Group'] == g1, 'Fitness']
            arr2 = data.loc[data['Group'] == g2, 'Fitness']
            stat, p_val = ranksums(arr1, arr2)
            star = significance_markers(p_val)

            print(f"{g1} vs {g2} | p={p_val:.4e} ({star})")

    # Save the figure
    plt.savefig(output_path / "bp_all_experiments.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    records = []
    for label, path in EXPERIMENT_PATHS.items():
        means = get_experiment_fitness_means(path)
        for m in means:
            records.append({'Group': label, 'Fitness': m})

    if not records:
        print("No data collected. Check your paths and data files.")
        return

    data = pd.DataFrame(records)
    create_multigroup_boxplot(data, Path("."))


if __name__ == "__main__":
    main()
