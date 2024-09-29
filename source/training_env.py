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

    def __str__(self):
        return f"RoughTerrain(floor_height={self.floor_height}, block_size={self.block_size})"


@dataclass(frozen=True)
class HillsTerrain:
    """Represents a hills terrain with a specific floor height and scale."""

    floor_height: float
    scale: int

    def __str__(self):
        return f"HillsTerrain(floor_height={self.floor_height}, scale={self.scale})"


@dataclass(frozen=True)
class DefaultTerrain:
    """Represents a default terrain with no specific attributes."""

    def __str__(self):
        return f"DefaultTerrain"

TerrainType = Union[RoughTerrain, HillsTerrain, DefaultTerrain]


class TrainingSchedule:
    """Manages the training and testing schedules for different terrain types."""

    hills_scale_range: tuple[int, int] = (5, 20)
    hills_scale_step: int = 5
    hills_floor_range: tuple[float, float] = (2.2, 4.0)
    hills_floor_step: float = 0.2

    rt_block_range: tuple[int, int] = (1, 4)
    rt_block_step: int = 1
    rt_floor_range: tuple[float, float] = (0.1, 1.0)
    rt_floor_step: float = 0.1

    def __init__(self):
        self.training_terrains: List[TerrainType] = self._init_training_terrains()
        self.testing_terrains = self._init_testing_terrains()
        self.all_terrains = self.training_terrains + self.testing_terrains

        self.store_training_terrains: List[TerrainType] = []

    def get_training_terrain(self, generation: int) -> TerrainType:
        """Retrieves a training terrain based on the generation number."""
        return self.training_terrains[generation % len(self.training_terrains)]

    def remove_training_terrain(self, index: int) -> TerrainType:
        """Removes and returns a training terrain from the list based on its index."""
        return self.training_terrains.pop(index)

    def setup_train_on_terrain_partition(self, p_terrains: List[TerrainType]):
        self.store_training_terrains = self.training_terrains
        self.training_terrains = p_terrains

    def restore_training_terrains(self):
        self.training_terrains = self.store_training_terrains

    def _init_training_terrains(self) -> List[TerrainType]:
        training_terrains_hills: List[HillsTerrain] = [
            HillsTerrain(2.6, 5),
            HillsTerrain(2.6, 10),
            HillsTerrain(2.6, 15),
            HillsTerrain(2.6, 20),
            HillsTerrain(2.8, 5),
            HillsTerrain(2.8, 10),
            HillsTerrain(2.8, 15),
            HillsTerrain(2.8, 20),
            HillsTerrain(3.4, 5),
            HillsTerrain(3.4, 10),
            HillsTerrain(3.4, 15),
            HillsTerrain(3.4, 20),
            HillsTerrain(3.6, 5),
            HillsTerrain(3.6, 10),
            HillsTerrain(3.6, 15),
            HillsTerrain(3.6, 20),
        ]
        training_terrains_rough: List[RoughTerrain] = [
            RoughTerrain(0.3, 1),
            RoughTerrain(0.3, 2),
            RoughTerrain(0.3, 3),
            RoughTerrain(0.3, 4),
            RoughTerrain(0.4, 1),
            RoughTerrain(0.4, 2),
            RoughTerrain(0.4, 3),
            RoughTerrain(0.4, 4),
            RoughTerrain(0.7, 1),
            RoughTerrain(0.7, 2),
            RoughTerrain(0.7, 3),
            RoughTerrain(0.7, 4),
            RoughTerrain(0.8, 1),
            RoughTerrain(0.8, 2),
            RoughTerrain(0.8, 3),
            RoughTerrain(0.8, 4),
        ]

        return [DefaultTerrain()] + training_terrains_hills + training_terrains_rough

    def _init_testing_terrains(self) -> List[TerrainType]:
        testing_terrains: List[TerrainType] = []

        for floor_height in np.round(
            np.arange(
                self.hills_floor_range[0],
                self.hills_floor_range[1] + self.hills_floor_step,
                self.hills_floor_step,
            ),
            1,
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
            ),
            1,
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
