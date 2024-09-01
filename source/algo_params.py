from abc import ABC

class AlgoParams(ABC):
    """Abstract base class to serve as a contract for subclasses."""
    pass

class AlgoParamsGeneralist(AlgoParams):
    init_training_generations: int = 2500
    max_generations: int = 5000
    gen_stagnation: int = 500

class AlgoParamsSpecialist(AlgoParams):
    init_training_generations: int = 2500
    max_generations: int = 10000
    gen_stagnation: int = 500