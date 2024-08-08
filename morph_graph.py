import argparse
import os
import torch
import pandas as pd
import numpy as np
from source.individual import Individual
import matplotlib.pyplot as plt

def create_plot(df, generations, ylabel, save_path):
    plt.figure(figsize=(20, 4))
    for column in df.columns:
        plt.plot(generations, df[column], label=column)

    plt.title(f"Ant {ylabel} Morphology Changes Over Generations")
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    # Setting x-axis ticks to show every generation mark if needed or skip some for clarity
    if len(generations) > 10:
        tick_spacing = int(len(generations) / 10)  # Shows a tick at every 10th generation if there are many points
        plt.xticks(generations[::tick_spacing])
    else:
        plt.xticks(generations)

    plt.xticks(rotation=45)  # Optional: rotate labels to improve readability
    plt.legend()
    plt.grid(True)  # Optional: adds grid lines for better readability
    plt.savefig(save_path)

parser = argparse.ArgumentParser(description="Create picture motion videos with a folder of images.")
parser.add_argument("--tensors_path", type=str, help="Path to the folder where the tensors of the generations are stored")
args = parser.parse_args()


tensors_path: str = args.tensors_path
ind: Individual = Individual(id=1234)

tensors = [tensor for tensor in os.listdir(tensors_path) if tensor.endswith(".pt")]

morph_data = []
for tensor in tensors:
    if tensor.endswith("best.pt"): break
    tensor_path = os.path.join(tensors_path, tensor)
    params = torch.load(tensor_path)
    ind.setup_ant_default(params)
    morph_data.append(ind.mjEnv.morphology.morph_params_map)

df = pd.DataFrame(morph_data)

width_columns = [col for col in df.columns if "width" in col]
length_columns = [col for col in df.columns if "length" in col]
aux_width_columns = [col for col in width_columns if "aux" in col]
ankle_width_columns = [col for col in width_columns if "ankle" in col]
aux_length_columns = [col for col in length_columns if "aux" in col]
ankle_length_columns = [col for col in length_columns if "ankle" in col]

df_aux_width = df[aux_width_columns]
df_ankle_width = df[ankle_width_columns]
df_aux_length = df[aux_length_columns]
df_ankle_length = df[ankle_length_columns]

generations = np.arange(10, 10 * len(df) + 10, 10)

create_plot(df_aux_width, generations, "Aux Leg Width",  "./aux_leg_width_plot.png")
create_plot(df_ankle_width, generations, "Ankle Leg Width",  "./ankle_leg_width_plot.png")
create_plot(df_aux_length, generations, "Aux Leg Length", "./aux_leg_length_plot.png")
create_plot(df_ankle_length, generations, "Ankle Leg Length", "./ankle_leg_length_plot.png")