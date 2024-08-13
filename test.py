import pickle
import argparse
from pathlib import Path

from source.training_env import TrainingSchedule
parser = argparse.ArgumentParser(
    description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch"
)
parser.add_argument(
    "--run_paths",
    nargs="+",
    type=Path,
    required=True,
    help="A list of paths to the run that you want to create combined graphs for.",
)
args = parser.parse_args()

g = []
e = []
for run_path in args.run_paths:
    with open(run_path / "G_var.pkl", "rb") as file:
        g.extend(pickle.load(file))

    with open(run_path / "E_var.pkl", "rb") as file:
        e.extend(pickle.load(file))

with open("./G_var.pkl", "wb") as file:
    pickle.dump(g, file)
with open("./E_var.pkl", "wb") as file:
    pickle.dump(e, file)
