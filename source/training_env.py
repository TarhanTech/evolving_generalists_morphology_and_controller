from source.globals import *
import numpy as np

class RoughTerrain:
    def __init__(self, floor_height: float, block_size: int):
        self.floor_height = floor_height
        self.block_size = block_size

class HillsTerrain:
    def __init__(self, floor_height: float, scale: int):
        self.floor_height = floor_height
        self.scale = scale

class DefaultTerrain: 
    def __init__(self):
        pass

class TrainingSchedule:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingSchedule, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.training_schedule = self._create_training_schedule()

    def _create_training_schedule(self):
        schedule = []
        schedule.append(DefaultTerrain())

        # Create and insert rough terrain environments
        for block_size in np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step): # (1, 2, 3, 4)
            for floor_height in np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step):
                schedule.append(RoughTerrain(floor_height, block_size))
        
        # Create and insert hill terrain environments
        for scale in np.arange(hills_scale_start, hills_scale_end + hills_scale_step, hills_scale_step):
            for floor_height in np.arange(hills_floor_start, hills_floor_end + hills_floor_step, hills_floor_step):
                schedule.append(HillsTerrain(floor_height, scale))

        return schedule
    
    def get_training_env(self, generation: int):
        return self.training_schedule[generation % len(self.training_schedule)]
    
    def remove_training_env(self, index: int):
        return self.training_schedule.pop(index)