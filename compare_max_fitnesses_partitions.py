import pandas as pd
from scipy import stats

def compare_max_fitness_columns(csv_file_1, csv_file_2, column_name):
    df1 = pd.read_csv(csv_file_1)
    df2 = pd.read_csv(csv_file_2)

    # Make sure both dataframes have the column
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Column '{column_name}' not found in both CSVs")

    # Drop any NaNs
    series1 = df1[column_name].dropna()
    series2 = df2[column_name].dropna()

    # Unpaired two-sample t-test
    t_stat, p_val = stats.ttest_ind(series1, series2, equal_var=False)

    # You can decide on your own significance threshold (p<0.05, etc.)
    print(f"Comparing {column_name}:")
    print(f"  Mean of {csv_file_1}: {series1.mean():.3f}")
    print(f"  Mean of {csv_file_2}: {series2.mean():.3f}")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.3g}\n")

def main():
    file_large = "morph_params_our_algo_largemorph.csv"
    file_morphevo = "morph_params_our_algo_morphevo.csv"

    columns_to_test = ["P1 max fitness", "P2 max fitness", "P3 max fitness"]

    for col in columns_to_test:
        compare_max_fitness_columns(file_large, file_morphevo, col)

if __name__ == "__main__":
    main()
