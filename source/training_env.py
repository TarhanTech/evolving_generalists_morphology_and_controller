"""
Module containing the different terrain environments and a training schedule for these terrains
"""

from dataclasses import dataclass
from typing import Union, List
import numpy as np


@dataclass(frozen=True)
class RoughTerrain:
    """Represents a rough terrain with a specific floor height and block size."""

    floor_height: float
    block_size: int


@dataclass(frozen=True)
class HillsTerrain:
    """Represents a hills terrain with a specific floor height and scale."""

    floor_height: float
    scale: int


@dataclass(frozen=True)
class DefaultTerrain:
    """Represents a default terrain with no specific attributes."""


TerrainType = Union[RoughTerrain, HillsTerrain, DefaultTerrain]


class TrainingSchedule:
    """Manages the training and testing schedules for different terrain types."""

    def __init__(self):
        self.hills_scale_range: tuple[int, int] = (5, 20)
        self.hills_scale_step: int = 5
        self.hills_floor_range: tuple[float, float] = (2.0, 4.6)
        self.hills_floor_step: float = 0.2

        self.rt_block_range: tuple[int, int] = (1, 4)
        self.rt_block_step: int = 1
        self.rt_floor_range: tuple[float, float] = (0.1, 1.4)
        self.rt_floor_step: float = 0.1

        self.training_terrains: List[TerrainType] = self._init_training_terrains()
        self.testing_terrains = self._init_testing_terrains()
        self.all_terrains = self.training_terrains + self.testing_terrains

    def get_training_terrain(self, generation: int) -> TerrainType:
        """Retrieves a training terrain based on the generation number."""
        return self.training_terrains[generation % len(self.training_terrains)]

    def remove_training_terrain(self, index: int) -> TerrainType:
        """Removes and returns a training terrain from the list based on its index."""
        return self.training_terrains.pop(index)

    def _init_training_terrains(self) -> List[TerrainType]:
        training_terrains_hills: List[HillsTerrain] = [
            HillsTerrain(2.4, 10),
            HillsTerrain(2.4, 15),
            HillsTerrain(2.6, 10),
            HillsTerrain(2.6, 15),
            HillsTerrain(3.2, 10),
            HillsTerrain(3.2, 15),
            HillsTerrain(3.4, 10),
            HillsTerrain(3.4, 15),
            HillsTerrain(4.0, 10),
            HillsTerrain(4.0, 15),
            HillsTerrain(4.2, 10),
            HillsTerrain(4.2, 15),
        ]
        training_terrains_rough: List[RoughTerrain] = [
            RoughTerrain(0.3, 2),
            RoughTerrain(0.3, 3),
            RoughTerrain(0.4, 2),
            RoughTerrain(0.4, 3),
            RoughTerrain(0.7, 2),
            RoughTerrain(0.7, 3),
            RoughTerrain(0.8, 2),
            RoughTerrain(0.8, 3),
            RoughTerrain(1.1, 2),
            RoughTerrain(1.1, 3),
            RoughTerrain(1.2, 2),
            RoughTerrain(1.2, 3),
        ]

        return training_terrains_hills + training_terrains_rough

    def _init_testing_terrains(self) -> List[TerrainType]:
        testing_terrains: List[TerrainType] = []
        testing_terrains.append(DefaultTerrain())

        for floor_height in np.round(
            np.arange(
                self.hills_floor_range[0],
                self.hills_floor_range[1] + self.hills_floor_step,
                self.hills_floor_step,
            )
        ):
            for scale in np.arange(
                self.hills_scale_range[0],
                self.hills_scale_range[1] + self.hills_scale_step,
                self.hills_scale_step,
            ):
                terrain: HillsTerrain = HillsTerrain(floor_height, scale)
                if terrain not in self.training_terrains:
                    testing_terrains.append(terrain)

        for floor_height in np.round(
            np.arange(
                self.rt_floor_range[0],
                self.rt_floor_range[1] + self.rt_floor_step,
                self.rt_floor_step,
            )
        ):
            for block_size in np.arange(
                self.rt_block_range[0],
                self.rt_block_range[1] + self.rt_block_step,
                self.rt_block_step,
            ):
                terrain: RoughTerrain = RoughTerrain(floor_height, block_size)
                if terrain not in self.training_terrains:
                    testing_terrains.append(terrain)

        return testing_terrains
