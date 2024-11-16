"""
Execute this file to run one of the experiments described in the paper.
Example: 'nohup python main.py experiment2 > experiment2.log 2>&1 &'
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from typing import List, Union
from pathlib import Path
import argparse
import os
import torch
from source.individual import Individual
from source.training_env import DefaultTerrain, HillsTerrain, RoughTerrain, TerrainType
from source.algo import FullGeneralist, OurAlgo, OurAlgoOneGen, Specialist


def our_algo(dis_morph_evo: bool, default_morph: bool):
    os.environ["MUJOCO_GL"] = "egl"
    algo: OurAlgo = OurAlgo(dis_morph_evo, default_morph, 23)
    algo.run()


def our_algo_one_gen():
    os.environ["MUJOCO_GL"] = "egl"
    algo: OurAlgoOneGen = OurAlgoOneGen(23)
    algo.run()

def full_gen(dis_morph_evo: bool):
    os.environ["MUJOCO_GL"] = "egl"
    algo: FullGeneralist = FullGeneralist(dis_morph_evo, 23)
    algo.run()

def specialist(dis_morph_evo: bool, long: bool):
    os.environ["MUJOCO_GL"] = "egl"
    algo: Specialist = Specialist(dis_morph_evo, long, 23)
    algo.run()


def test_ant(tensor_path: Path, terrain: str, params: str):
    """Runs a test"""

    def convert_param(param: str) -> Union[int, float]:
        """Convert a string parameter to int or float."""
        try:
            return int(param)
        except ValueError:
            try:
                return float(param)
            except ValueError as e:
                raise ValueError(
                    f"Parameter '{param}' is not a valid integer or float."
                ) from e

    terrain_type: TerrainType = TERRAIN_MAP[terrain]
    ind: Individual = Individual("cpu", (-0.1, 0.1), 1.03, 100, 1000)
    tensor_params = torch.load(tensor_path)
    if terrain_type is DefaultTerrain:
        ind.setup_ant_default(tensor_params)
    elif terrain_type is HillsTerrain:
        params: List[Union[int, float]] = [convert_param(p) for p in params]
        ind.setup_ant_hills(tensor_params, params[0], params[1])
    elif terrain_type is RoughTerrain:
        params: List[Union[int, float]] = [convert_param(p) for p in params]
        ind.setup_ant_rough(tensor_params, params[0], params[1])
    else:
        assert False, "Class type not supported"

    total_reward: float = ind.evaluate_fitness(render_mode="human")
    print(f"Total Rewards: {total_reward}")


TERRAIN_MAP = {
    "default": DefaultTerrain,
    "hills": HillsTerrain,
    "rough": RoughTerrain,
}


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(
        dest="experiment", required=True, help="Choose an experiment to run."
    )

    parser_our_algo = subparsers.add_parser(
        "our_algo",
        help="Run the experiment where you create a generalist for each partition of the environments.",
    )
    parser_our_algo.add_argument(
        "--dis_morph_evo",
        action="store_true",
        default=False,
    )
    parser_our_algo.add_argument(
        "--default_morph",
        action="store_true",
        default=False,
    )

    parser_our_algo_one_gen = subparsers.add_parser(
        "our_algo_one_gen",
        help="Run the experiment where you create one generalist for all the environments",
    )

    parser_full_gen = subparsers.add_parser("full_gen")
    parser_full_gen.add_argument(
        "--dis_morph_evo",
        action="store_true",
        default=False,
    )

    parser_specialist = subparsers.add_parser(
        "specialist",
        help="Run the fourth experiment where you create a specialist for each of the environments using same resources as experiment 1",
    )
    parser_specialist.add_argument(
        "--dis_morph_evo",
        action="store_true",
        default=False,
    )
    parser_specialist.add_argument(
        "--long",
        action="store_true",
        default=False,
    )

    parser_test = subparsers.add_parser(
        "test",
        help="Run a test experiment to visual a generated tensor from the tensor folders",
    )
    parser_test.add_argument(
        "--tensor",
        type=Path,
        required=True,
        help="Path to the tensor that you want to test.",
    )
    parser_test.add_argument(
        "--terrain",
        type=str,
        choices=TERRAIN_MAP.keys(),
        required=True,
        help="Terrain environment to test in",
    )
    parser_test.add_argument(
        "--params",
        type=str,
        nargs="+",
        required=False,
        help="A list of parameters for the selected class. Parameters can be integers or floats.",
    )

    args = parser.parse_args()

    if args.experiment == "our_algo":
        our_algo(args.dis_morph_evo, args.default_morph)
    elif args.experiment == "our_algo_one_gen":
        our_algo_one_gen()
    elif args.experiment == "full_gen":
        full_gen(args.dis_morph_evo)
    elif args.experiment == "specialist":
        specialist(args.dis_morph_evo, args.long)
    elif args.experiment == "test":
        test_ant(args.tensor, args.terrain, args.params)
    else:
        print("Something went wrong!")


if __name__ == "__main__":
    main()
