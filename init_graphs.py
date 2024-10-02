import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
from source.mj_env import Morphology
def _decode_morph_params(morph_params: Tensor) -> Tensor:
    length_params, width_params = torch.split(
        morph_params, (Morphology.total_leg_length_params, Morphology.total_leg_width_params)
    )

    a: float = (Morphology.leg_length_range[1] - Morphology.leg_length_range[0]) / (
        Morphology.morph_params_bounds_enc[1] - Morphology.morph_params_bounds_enc[0]
    )
    b: float = Morphology.leg_length_range[0] - (a * Morphology.morph_params_bounds_enc[0])
    decoded_leg_length: Tensor = (a * length_params) + b

    c: float = (Morphology.leg_width_range[1] - Morphology.leg_width_range[0]) / (
        Morphology.morph_params_bounds_enc[1] - Morphology.morph_params_bounds_enc[0]
    )
    d: float = Morphology.leg_width_range[0] - (c * Morphology.morph_params_bounds_enc[0])
    decoded_leg_width: Tensor = (c * width_params) + d

    return torch.cat((decoded_leg_length, decoded_leg_width), dim=0)

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