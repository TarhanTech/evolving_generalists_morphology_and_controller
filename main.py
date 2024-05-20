from individual import *
import evotorch
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch import Problem

class AntProblem(Problem):
    def __init__(self, ind: Individual):
        nn_lower_bounds = [-0.00001] * ind.controller.total_weigths
        morph_leg_length_lower_bounds = [ind.morphology.leg_length_range[0]] * 8
        morph_leg_width_lower_bounds = [ind.morphology.leg_width_range[0]] * 8
        lower_bounds = nn_lower_bounds + morph_leg_length_lower_bounds + morph_leg_width_lower_bounds

        nn_upper_bounds = [0.00001] * ind.controller.total_weigths
        morph_leg_length_upper_bounds = [ind.morphology.leg_length_range[0]] * 8
        morph_leg_width_upper_bounds = [ind.morphology.leg_width_range[0]] * 8
        upper_bounds = nn_upper_bounds + morph_leg_length_upper_bounds + morph_leg_width_upper_bounds
        
        super().__init__("max", solution_length=ind.params_size, bounds=(lower_bounds, upper_bounds), dtype=torch.float64, eval_dtype=torch.float64, device="cuda")
        self.ind: Individual = ind
        
    def _evaluate(self, solution: evotorch.Solution):
        x = solution.values
        nn_params, morph_params = torch.split(x, (self.ind.controller.total_weigths, self.ind.morphology.total_genes))
        self.ind.morphology.set_morph_params(morph_params)
        self.ind.controller.set_nn_params(nn_params)
        solution.set_evals(self.ind.evaluate_fitness())
        

def main():
    ind: Individual = Individual()
    ind.print_morph_info()
    ind.print_controller_info()

    problem : AntProblem = AntProblem(ind=ind)
    searcher: XNES = XNES(problem, stdev_init=0.01)

    stdout_logger: StdOutLogger = StdOutLogger(searcher)
    pandas_logger: PandasLogger = PandasLogger(searcher)
    searcher.run(2500)
    
    pop_best_solution = searcher.status["pop_best"].values
    torch.save(pop_best_solution, "pop_best.pt")
    nn_params, morph_params = torch.split(pop_best_solution, (ind.controller.total_weigths, ind.morphology.total_genes))
    ind.morphology.set_morph_params(morph_params)
    ind.controller.set_nn_params(nn_params)

    df = pandas_logger.to_dataframe()
    df.to_csv("pandas_df.csv", index=False)

    ind.evaluate_fitness_rendered()

    print(f"pop_best.pt: {torch.load('pop_best.pt')}")


if __name__ == "__main__": main()