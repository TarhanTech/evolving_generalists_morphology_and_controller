import pandas as pd
import statsmodels.api as sm

def analyze_lengths_vs_fitness(df: pd.DataFrame, max_partition=4):
    results = {}
    for p in range(1, max_partition + 1):
        fitness_col = f"P{p} max fitness"
        length_cols = [
            c for c in df.columns 
            if c.startswith(f"P{p}") and c.endswith("_length")
        ]
        if fitness_col in df.columns and length_cols:
            subdf = df[length_cols + [fitness_col]].dropna()
            if subdf.empty:
                continue
            X = subdf[length_cols]
            y = subdf[fitness_col]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            results[f"P{p}"] = model
    return results

def main():
    df = pd.read_csv("morph_params_our_algo_morphevo.csv")
    analysis_results = analyze_lengths_vs_fitness(df, max_partition=3)
    for partition, model in analysis_results.items():
        print(partition)
        print(model.summary())

if __name__ == "__main__":
    main()
