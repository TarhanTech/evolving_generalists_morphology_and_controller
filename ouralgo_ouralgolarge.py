from source.training_env import DefaultTerrain, HillsTerrain, RoughTerrain, TerrainType
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu

def load_env_fitnesses(folder):
    env_to_fitnesses = defaultdict(list)
    for filename in ("env_fitnesses_test.pkl", "env_fitnesses_training.pkl"):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = pickle.load(f)
            for terrain_obj, fitness in data:
                env_to_fitnesses[terrain_obj.short_string()].append(fitness)
    return env_to_fitnesses

def aggregate_runs(base_dir):
    aggregated = defaultdict(list)
    if not os.path.isdir(base_dir):
        return aggregated
    run_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for subfolder in run_folders:
        folder_path = os.path.join(base_dir, subfolder)
        run_data = load_env_fitnesses(folder_path)
        for k, v in run_data.items():
            aggregated[k].extend(v)
    return aggregated

def create_empty_dataframes_for_terrains():
    hf_hills = np.round(np.arange(2.2, 4.1, 0.2), 1)
    sc_hills = [5, 10, 15, 20]
    hills_df = pd.DataFrame(np.nan, index=sc_hills, columns=hf_hills)

    hf_rough = np.round(np.arange(0.1, 1.1, 0.1), 1)
    bs_rough = [1, 2, 3, 4]
    rt_df = pd.DataFrame(np.nan, index=bs_rough, columns=hf_rough)

    default_df = pd.DataFrame([[np.nan]], index=[0], columns=[0])
    return default_df, hills_df, rt_df

def generate_single_pdf_heatmap(medians, output_pdf, testing_terrains=None):
    default_df, hills_df, rt_df = create_empty_dataframes_for_terrains()
    for k, fitness in medians.items():
        if k == "def":
            default_df.iloc[0, 0] = fitness
        elif k.startswith("ht("):
            inside = k[3:-1]
            fs, ss = inside.split(",")
            fh = float(fs)
            sc = float(ss)
            if sc in hills_df.index and fh in hills_df.columns:
                hills_df.loc[sc, fh] = fitness
        elif k.startswith("rt("):
            inside = k[3:-1]
            fs, bs = inside.split(",")
            fh = float(fs)
            b = float(bs)
            if b in rt_df.index and fh in rt_df.columns:
                rt_df.loc[b, fh] = fitness

    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])
    vmin, vmax = 0, 2500
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    sns.heatmap(hills_df, ax=ax0, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax0.set_title("Hill Environment")
    ax0.set_xlabel("Floor Height")
    ax0.set_ylabel("Scale")

    if testing_terrains:
        for t in testing_terrains:
            if isinstance(t, HillsTerrain):
                if t.scale in hills_df.index and round(t.floor_height, 1) in hills_df.columns:
                    ri = hills_df.index.get_loc(t.scale)
                    ci = hills_df.columns.get_loc(round(t.floor_height, 1))
                    ax0.add_patch(Rectangle((ci, ri), 1, 1, fill=False, edgecolor="red", lw=3))

    sns.heatmap(rt_df, ax=ax1, annot=True, cbar=False, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax1.set_title("Rough Environment")
    ax1.set_xlabel("Floor Height")
    ax1.set_ylabel("Block Size")

    if testing_terrains:
        for t in testing_terrains:
            if isinstance(t, RoughTerrain):
                if t.block_size in rt_df.index and round(t.floor_height, 1) in rt_df.columns:
                    ri = rt_df.index.get_loc(t.block_size)
                    ci = rt_df.columns.get_loc(round(t.floor_height, 1))
                    ax1.add_patch(Rectangle((ci, ri), 1, 1, fill=False, edgecolor="red", lw=3))

    sns.heatmap(default_df, ax=ax2, annot=True, cbar=True, cmap="gray", vmin=vmin, vmax=vmax, fmt=".1f")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title("Default Environment")

    vals = []
    for df in [hills_df, rt_df, default_df]:
        v = df.values[~np.isnan(df.values)]
        vals.extend(v)

    if len(vals) > 0:
        m = np.mean(vals)
        s = np.std(vals)
        plt.figtext(0.5, -0.02, f"Overall Mean: {m:.2f}, Overall STD: {s:.2f}", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close()

def generate_comparison_pdf_heatmap(data_a, data_b, output_pdf, name_a="A", name_b="B"):
    default_df, hills_df, rt_df = create_empty_dataframes_for_terrains()

    compare_hills = pd.DataFrame("", index=hills_df.index, columns=hills_df.columns)
    compare_rt = pd.DataFrame("", index=rt_df.index, columns=rt_df.columns)
    compare_default = pd.DataFrame([[""]], index=[0], columns=[0])

    num_hills = pd.DataFrame(0, index=hills_df.index, columns=hills_df.columns)
    num_rt = pd.DataFrame(0, index=rt_df.index, columns=rt_df.columns)
    num_default = pd.DataFrame([[0]], index=[0], columns=[0])

    all_keys = set(data_a.keys()) | set(data_b.keys())

    for env_key in all_keys:
        if env_key in data_a and env_key in data_b:
            vals_a = data_a[env_key]
            vals_b = data_b[env_key]
            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")

            if p < 0.001:
                level = 3
            elif p < 0.01:
                level = 2
            elif p < 0.05:
                level = 1
            else:
                level = 0

            med_a = np.median(vals_a)
            med_b = np.median(vals_b)

            if level == 0:
                symbol = "="
            else:
                symbol = "-" * level if med_a > med_b else "+" * level

            if env_key == "def":
                compare_default.iloc[0, 0] = symbol
            elif env_key.startswith("ht("):
                inside = env_key[3:-1]
                fs, ss = inside.split(",")
                fh = float(fs)
                sc = float(ss)
                if sc in compare_hills.index and fh in compare_hills.columns:
                    compare_hills.loc[sc, fh] = symbol
            elif env_key.startswith("rt("):
                inside = env_key[3:-1]
                fs, bs = inside.split(",")
                fh = float(fs)
                b = float(bs)
                if b in compare_rt.index and fh in compare_rt.columns:
                    compare_rt.loc[b, fh] = symbol

    # Convert symbols to numeric values for coloring
    for i in compare_hills.index:
        for c in compare_hills.columns:
            symb = compare_hills.loc[i, c]
            if symb == "=":
                val = 0
            elif symb.startswith("+"):
                val = len(symb)
            elif symb.startswith("-"):
                val = -len(symb)
            else:
                val = 0
            num_hills.loc[i, c] = val

    for i in compare_rt.index:
        for c in compare_rt.columns:
            symb = compare_rt.loc[i, c]
            if symb == "=":
                val = 0
            elif symb.startswith("+"):
                val = len(symb)
            elif symb.startswith("-"):
                val = -len(symb)
            else:
                val = 0
            num_rt.loc[i, c] = val

    d_symb = compare_default.iloc[0, 0]
    if d_symb == "=":
        d_val = 0
    elif d_symb.startswith("+"):
        d_val = len(d_symb)
    elif d_symb.startswith("-"):
        d_val = -len(d_symb)
    else:
        d_val = 0
    num_default.iloc[0, 0] = d_val

    plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    sns.heatmap(num_hills, ax=ax0, annot=compare_hills, cbar=False, cmap="bwr", vmin=-3, vmax=3, fmt="")
    ax0.set_title(f"Hills Comparison")
    ax0.set_xlabel("Floor Height")
    ax0.set_ylabel("Scale")

    sns.heatmap(num_rt, ax=ax1, annot=compare_rt, cbar=False, cmap="bwr", vmin=-3, vmax=3, fmt="")
    ax1.set_title(f"Rough Comparison")
    ax1.set_xlabel("Floor Height")
    ax1.set_ylabel("Block Size")

    sns.heatmap(num_default, ax=ax2, annot=compare_default, cbar=True, cmap="bwr", vmin=-3, vmax=3, fmt="")
    ax2.set_title(f"Default Comparison")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    output_folder = "./_graphs_for_paper"
    os.makedirs(output_folder, exist_ok=True)

    # OurAlgo-LargeMorph-Gen
    large_dir = "./runs/OurAlgo-LargeMorph-Gen"
    large_data = aggregate_runs(large_dir)
    large_medians = {k: np.median(v) for k, v in large_data.items() if len(v) > 0}
    large_pdf = os.path.join(output_folder, "OurAlgo-LargeMorph-Gen_fitness_heatmap.pdf")
    generate_single_pdf_heatmap(large_medians, large_pdf)

    # OurAlgo-MorphEvo-Gen
    morph_dir = "./runs/OurAlgo-MorphEvo-Gen"
    morph_data = aggregate_runs(morph_dir)
    morph_medians = {k: np.median(v) for k, v in morph_data.items() if len(v) > 0}
    morph_pdf = os.path.join(output_folder, "OurAlgo-MorphEvo-Gen_fitness_heatmap.pdf")
    generate_single_pdf_heatmap(morph_medians, morph_pdf)

    # Comparison: OurAlgo-LargeMorph-Gen vs OurAlgo-MorphEvo-Gen
    comp_pdf = os.path.join(output_folder, "comparison_Large_vs_MorphEvo.pdf")
    generate_comparison_pdf_heatmap(
        data_a=large_data,
        data_b=morph_data,
        output_pdf=comp_pdf,
        name_a="LargeMorph",
        name_b="MorphEvo"
    )

    print("Saved heatmaps to:", output_folder)
