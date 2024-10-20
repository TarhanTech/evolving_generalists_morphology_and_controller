import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
from source.mj_env import Morphology
from source.individual import Individual
import argparse
from pathlib import Path
import os


def get_creation_time(tensor_file):
    return os.path.getctime(args.run_path / tensor_file)

parser = argparse.ArgumentParser(
    description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch"
)
parser.add_argument(
    "--run_path",
    type=Path,
    required=True,
    help="A path of generalist tensors",
)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ind = Individual(
    device=device, morph_params_bounds_enc=(-0.1, 0.1), penalty_growth_rate=1.03, penalty_scale_factor=100, penalty_scale_factor_err=1000
)
sorted_tensor_files = sorted(os.listdir(args.run_path), key=get_creation_time)
morph_data = []
for i, tensor_file in enumerate(sorted_tensor_files):
    if i % 10 == 0 or tensor_file.endswith("best.pt"):
        tensor_path = args.run_path / tensor_file
        params = torch.load(tensor_path)
        ind.setup_ant_default(params)

        morph_data.append({
            **ind.mj_env.morphology.morph_params_map,
            "Generation": i + 1
        })
morph_data_df = pd.DataFrame(morph_data).set_index("Generation")
pd.DataFrame(morph_data_df).to_csv("morph_data_df.csv", index=False)

similarities = cosine_similarity(morph_data_df.values)

similarity_df = pd.DataFrame(similarities)

similarity_df.to_csv("./similarity_matrix.csv", index=False)

edge_list = []
for i in range(similarity_df.shape[0]):
    for j in range(i + 1, similarity_df.shape[1]):
        weight = similarity_df.iloc[i, j]
        if weight > 0.99:
            edge_list.append((i, j, weight))

edges_df = pd.DataFrame(edge_list, columns=["Source", "Target", "Weight"])

edges_df.to_csv("./gephi_edges.csv", index=False)


import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Louvain algorithm

# Load the edge list from your CSV
edges_df = pd.read_csv("./gephi_edges.csv")

# Create a graph from the edge list
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(int(row["Source"]), int(row["Target"]), weight=row["Weight"])

# Apply Louvain community detection
partition = community_louvain.best_partition(G, weight='weight')

# Draw the graph with the node colors based on cluster membership
pos = nx.spring_layout(G)  # Use spring layout for a visually appealing arrangement
plt.figure(figsize=(12, 12))

# Color nodes by their cluster (partition value)
cmap = plt.get_cmap('viridis')
colors = [cmap(partition[node]) for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.5)

plt.title("Graph Clustering with Louvain Algorithm")

# Save the plot as a PDF
plt.savefig("graph_clustering.pdf", format="pdf")
