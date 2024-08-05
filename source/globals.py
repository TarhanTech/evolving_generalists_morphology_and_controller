# Folders
train_ant_xml_folder: str = "train_ant_xml"
train_terrain_noise_folder: str = "train_terrain_noise"

# Algorithm Parameters
algo_stdev_init = 0.01
algo_params_range = (-0.1, 0.1)
algo_init_training_generations: int = 2500
penalty_growth_rate = 1.03
penalty_scale_factor: int = 100
penalty_scale_factor_err: int = 1000

# Generalist Training Parameters
algo_max_generations: int = 5000
algo_gen_stagnation: int = 500

# Specialist Training Parameters
spec_algo_max_generations: int = 10000
spec_algo_gen_stagnation: int = 200

# Training and Validation schedule
# Rough Terrain
rt_block_start: int = 1
rt_block_end: int = 4
rt_block_step: int = 1
rt_floor_start: float = 0.1
rt_floor_end: float = 1.0
rt_floor_step: float = 0.1
#Hills Terrain
hills_scale_start: int = 5
hills_scale_end: int = 20
hills_scale_step: int = 5
hills_floor_start: float = 2
hills_floor_end: float = 4
hills_floor_step: float = 0.2