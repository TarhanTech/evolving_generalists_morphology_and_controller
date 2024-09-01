"""This executable creates graphs, images and videos for the experimental runs"""

from pathlib import Path
import argparse
from typing import List
import matplotlib.pyplot as plt

from source.globals import *
from source.utils.graph_builder import Graphbuilder, GraphBuilderGeneralist, GraphBuilderSpecialist, GraphBuilderCombination

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """
    Initial entry point of the executable.
    Execute:
    python graph.py --run_paths path1 path2 path3
    """
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
    parser.add_argument(
        "--specialist",
        nargs="+",
        type=str2bool,
        default=[False],
        help="Pass in true to create graphs for a specialist run, leave empty for generalist",
    )
    args = parser.parse_args()
    print(args.specialist)
    if len(args.run_paths) != len(args.specialist):
        parser.error("run_paths and specialist must have the same number of elements.")

    if len(args.run_paths) == 1:
        if args.specialist[0] is True:
            graph_builder_spec: GraphBuilderSpecialist = GraphBuilderSpecialist(args.run_paths[0])
            graph_builder_spec.create_ant_screenshots()
            graph_builder_spec.create_generalist_heatmap_partition()
            graph_builder_spec.create_fitness_heatmap()
            graph_builder_spec.create_fitness_env_boxplot()

            graph_builder_spec.create_fitness_evaluation_graph()
            # graph_builder_spec.create_morph_params_plot()
            # graph_builder_spec.create_morph_params_pca_scatterplot()
            graph_builder_spec.create_evolution_video()
        elif args.specialist[0] is False:
            graph_builder_gen: GraphBuilderGeneralist = GraphBuilderGeneralist(args.run_paths[0])
            graph_builder_gen.create_ant_screenshots()
            graph_builder_gen.create_generalist_heatmap_partition()
            graph_builder_gen.create_fitness_heatmap()
            graph_builder_gen.create_fitness_env_boxplot()

            graph_builder_gen.create_generalist_evaluation_graph()
            graph_builder_gen.create_morph_params_plot()
            graph_builder_gen.create_morph_params_pca_scatterplot()
            graph_builder_gen.create_evolution_video()
        else:
            assert False, "Unknown error has occurred"
    else:
        gbs: List[Graphbuilder] = []
        for run_path in args.run_paths:
            gb: Graphbuilder = Graphbuilder(run_path)
            gbs.append(gb)

        graph_builder_comb = GraphBuilderCombination(gbs)
        graph_builder_comb.create_boxplot_experiments()

    plt.close()


if __name__ == "__main__":
    main()
