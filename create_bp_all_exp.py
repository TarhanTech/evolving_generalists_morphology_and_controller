from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import kruskal
import pandas as pd
import scikit_posthocs as sp

# Hardcoded base runs path
BASE_RUNS_PATH = Path("./runs")

EXPERIMENT_PATHS = {
    # "FullGen-DefaultMorph-Gen": BASE_RUNS_PATH / "FullGen-DefaultMorph-Gen",
    "FullGen-MorphEvo-Gen": BASE_RUNS_PATH / "FullGen-MorphEvo-Gen",
    # "OurAlgo-DefaultMorph-Gen": BASE_RUNS_PATH / "OurAlgo-DefaultMorph-Gen",
    "OurAlgo-LargeMorph-Gen": BASE_RUNS_PATH / "OurAlgo-LargeMorph-Gen",
    "OurAlgo-MorphEvo-Gen": BASE_RUNS_PATH / "OurAlgo-MorphEvo-Gen",
    # "Spec-DefaultMorph": BASE_RUNS_PATH / "Spec-DefaultMorph",
    # "Spec-MorphEvo": BASE_RUNS_PATH / "Spec-MorphEvo",
    # "Spec-MorphEvo-Long": BASE_RUNS_PATH / "Spec-MorphEvo-Long"
}


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
    return fitness_means


def significance_markers(p):
    if p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'


def add_significance_annotations(ax, data: pd.DataFrame, dunn_results: pd.DataFrame):
    """
    Annotate the plot with significance lines for each pair that is significant (p < 0.05).
    Using the original spacing logic.
    """
    groups = list(data['Group'].unique())
    sig_pairs = []
    for i, g1 in enumerate(groups):
        for j in range(i+1, len(groups)):
            g2 = groups[j]
            p = dunn_results.loc[g1, g2]
            if p < 0.05:
                sig_pairs.append((g1, g2, p))

    if not sig_pairs:
        return  # No significant pairs, nothing to annotate

    max_val = data['Fitness'].max()
    min_val = data['Fitness'].min()

    # Original spacing
    step = (max_val - min_val) * 0.05  
    height = max_val + step

    # Sort pairs by p-value
    sig_pairs = sorted(sig_pairs, key=lambda x: x[2])

    for (g1, g2, p) in sig_pairs:
        x1 = groups.index(g1)
        x2 = groups.index(g2)
        if x1 > x2:
            x1, x2 = x2, x1

        line_x = [x1, x1, x2, x2]
        line_y = [height, height + step * 0.3, height + step * 0.3, height]
        ax.plot(line_x, line_y, c='k', lw=1.5)
        ax.text((x1 + x2)*0.5, height + step * 0.3, significance_markers(p),
                ha='center', va='bottom', color='k', fontsize=10)

        height += step


def create_multigroup_boxplot(data: pd.DataFrame, output_path: Path):
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    groups = data['Group'].unique()
    order = list(groups)

    palette = sns.color_palette("tab10", n_colors=len(groups))
    palette = dict(zip(groups, palette))

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

    sns.stripplot(
        x='Group',
        y='Fitness',
        data=data,
        order=order,
        palette=palette,
        size=5,
        jitter=True,
        dodge=False
    )

    boxplot.set_title("Comparison of All Experiments", fontsize=14, fontweight="bold")
    boxplot.set_ylabel("Fitness", fontsize=12)
    boxplot.set_xlabel("Experiments", fontsize=12)
    boxplot.tick_params(labelsize=10)

    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)

    # Perform Kruskal-Wallis test for overall difference
    group_data = [data.loc[data['Group'] == g, 'Fitness'] for g in order]
    stat, p_value_kruskal = kruskal(*group_data)
    print(f"Kruskal-Wallis test: H={stat:.3f}, p={p_value_kruskal:.3e}")

    # Perform Dunn's posthoc test
    dunn_results = sp.posthoc_dunn(data, val_col='Fitness', group_col='Group', p_adjust='bonferroni')
    print("Dunn's post-hoc test results (p-values):")
    print(dunn_results)

    # Print out all pairs with their p-values and significance
    print("\nPairwise comparisons:")
    for i, g1 in enumerate(order):
        for j in range(i+1, len(order)):
            g2 = order[j]
            p = dunn_results.loc[g1, g2]
            print(f"{g1} vs {g2}: p={p:.4e} ({significance_markers(p)})")

    # Add significance annotations for all significant pairs
    # add_significance_annotations(boxplot, data, dunn_results)

    # Save figure
    plt.savefig(output_path / "bp_all_experiments.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    records = []
    for label, path in EXPERIMENT_PATHS.items():
        means = get_experiment_fitness_means(path)
        # if label == "Spec-MorphEvo-Long":
        #     means = (means * (30 // len(means) + 1))[:30]
        #     print(means)
        for m in means:
            records.append({'Group': label, 'Fitness': m})

    if not records:
        print("No data collected. Check your paths and data files.")
        return

    data = pd.DataFrame(records)
    create_multigroup_boxplot(data, Path("."))


if __name__ == "__main__":
    main()
