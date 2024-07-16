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
        self.floor_heights_for_testing_rough = [0.1, 0.5, 0.6, 1.0]
        self.floor_heights_for_testing_hills = [2.0, 2.8, 3.0, 3.2, 4.0]
        self.training_schedule = []
        self.testing_schedule = []
        self.total_schedule = []
        self._init_schedules()

    def _init_schedules(self):
        self.training_schedule.append(DefaultTerrain())

        # Create and insert rough terrain environments
        for block_size in np.arange(rt_block_start, rt_block_end + rt_block_step, rt_block_step): # (1, 2, 3, 4)
            for floor_height in np.arange(rt_floor_start, rt_floor_end + rt_floor_step, rt_floor_step):
                floor_height_rounded = round(floor_height, 1)
                if floor_height_rounded in self.floor_heights_for_testing_rough:
                    self.testing_schedule.append(RoughTerrain(floor_height_rounded, block_size))
                else:
                    self.training_schedule.append(RoughTerrain(floor_height_rounded, block_size))
        
        # Create and insert hill terrain environments
        for scale in np.arange(hills_scale_start, hills_scale_end + hills_scale_step, hills_scale_step):
            for floor_height in np.arange(hills_floor_start, hills_floor_end + hills_floor_step, hills_floor_step):
                floor_height_rounded = round(floor_height, 1)
                if floor_height_rounded in self.floor_heights_for_testing_hills:
                    self.testing_schedule.append(HillsTerrain(floor_height_rounded, scale))
                else:
                    self.training_schedule.append(HillsTerrain(floor_height_rounded, scale))
        
        self.total_schedule = self.training_schedule + self.testing_schedule
    
    def get_training_env(self, generation: int):
        return self.training_schedule[generation % len(self.training_schedule)]
    
    def remove_training_env(self, index: int):
        return self.training_schedule.pop(index)