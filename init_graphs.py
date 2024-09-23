import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('./gen_score_pandas_df.csv')

similarities = cosine_similarity(df.values)

similarity_df = pd.DataFrame(similarities)

similarity_df.to_csv('./similarity_matrix.csv', index=False)

edge_list = []
for i in range(similarity_df.shape[0]):
    for j in range(i+1, similarity_df.shape[1]):
        weight = similarity_df.iloc[i, j]
        if weight > 0.99:
            edge_list.append((i, j, weight))

edges_df = pd.DataFrame(edge_list, columns=['Source', 'Target', 'Weight'])

edges_df.to_csv('./gephi_edges.csv', index=False)