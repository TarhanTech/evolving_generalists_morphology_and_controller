"""This executable creates graphs, images and videos for the experimental runs"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from source.graph_builder import (
    GraphBuilderGeneralist,
    GraphBuilderSpecialist,
    GraphBuilderCombination,
)

os.environ["MUJOCO_GL"] = "egl" # This line is to ensure that the script also works on a server.

def main():
    """
    Initial entry point of the executable.
    """
    parser = argparse.ArgumentParser(
        description="Evolving generalist controller and morphology to handle wide range of environments. Run script without arguments to train an ant from scratch"
    )
    subparsers = parser.add_subparsers(
        dest="type", required=True, help="Choose what type of MC-pair was created for the run."
    )

    parser_generalist = subparsers.add_parser(
        "generalist", help="Create graphs meant for a run path of generalist."
    )
    parser_generalist.add_argument(
        "--run_path",
        type=Path,
        required=True,
        help="A path to the run of a generalist experimental run.",
    )
    parser_generalist.add_argument(
        "--videos",
        action="store_true",
        default=False,
        help="CReate videos"
    )
    parser_generalist.add_argument(
        "--dis_morph_evo",
        action="store_true",
        default=False,
        help=""
    )
    parser_generalist.add_argument(
        "--morph_type",
        type=str,
    )

    parser_specialist = subparsers.add_parser(
        "specialist", help="Create graphs meant for a run path of specialist."
    )
    parser_specialist.add_argument(
        "--run_path",
        type=Path,
        required=True,
        help="A path to the run of a specialist experimental run.",
    )
    parser_specialist.add_argument(
        "--videos",
        action="store_true",
        default=False,
        help="Enable verbose output."
    )
    parser_specialist.add_argument(
        "--dis_morph_evo",
        action="store_true",
        default=False,
        help=""
    )
    parser_specialist.add_argument(
        "--morph_type",
        type=str,
    )

    parser_combined = subparsers.add_parser("combined", help="Create combined graphs to compare results from different experiments.")
    parser_combined.add_argument(
        "--run_paths",
        nargs="+",
        type=Path,
        help="A list of paths to multiple experimental runs to create combined graphs to compare the results.",
    )

    args = parser.parse_args()
                
    if args.type == "generalist":
        graph_builder_gen: GraphBuilderGeneralist = GraphBuilderGeneralist(args.run_path, args.videos, args.dis_morph_evo, args.morph_type)
        graph_builder_gen.create_graphs()
        
    elif args.type == "specialist":
        graph_builder_spec: GraphBuilderSpecialist = GraphBuilderSpecialist(args.run_path, args.videos, args.dis_morph_evo, args.morph_type)
        graph_builder_spec.create_graphs()
    elif args.type == "combined":
        gs = GraphBuilderCombination(args.run_paths)
        gs.create_graphs()

    print("Creating graphs has finished!")
    plt.close()


if __name__ == "__main__":
    main()
