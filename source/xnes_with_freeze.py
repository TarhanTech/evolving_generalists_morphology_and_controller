import torch
from evotorch.algorithms.distributed.gaussian import XNES
from evotorch.core import SolutionBatch

class XNESWithFreeze(XNES):
    def __init__(
        self,
        freeze_interval,
        freeze_params: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.increase_interval_activated = False
        if freeze_interval == 0:
            self.increase_interval_activated = True

        if freeze_params is None:
            raise ValueError(
                "freeze_params cannot be None. "
                "Use the standard XNES if you don't want to freeze any parameters."
            )

        self.freeze_params = freeze_params
        self.freeze_interval = freeze_interval

        self.nn_params_size = self.problem.nn_params_size
        self.morph_params_size = self.problem.morph_params_size

        self.morph_start = self.nn_params_size
        self.morph_end = self.nn_params_size + self.morph_params_size

        self._stored_params = None

    def _step_non_distributed(self):
        if (self.step_count % 100) == 0 and self.increase_interval_activated:
            self.freeze_interval = self.freeze_interval + 1

        if self._population is None:
            self._population = SolutionBatch(
                self.problem,
                popsize=self._popsize,
                device=self._distribution.device,
                empty=True
            )
            self._distribution.sample(
                out=self._population.access_values(),
                generator=self.problem
            )
            self.problem.evaluate(self._population)

            pop_values = self._population.access_values(keep_evals=True)
            if self.freeze_params == "morphology":
                self._stored_params = pop_values[:, self.morph_start : self.morph_end].clone()
            elif self.freeze_params == "controller":
                self._stored_params = pop_values[:, 0 : self.nn_params_size].clone()
            else:
                raise ValueError("freeze_params must be 'morphology' or 'controller'")

            self._first_iter = False
            return

        old_values = self._population.access_values(keep_evals=True)
        old_fitnesses = self._population.access_evals()[:, self._obj_index]

        gradients = self._distribution.compute_gradients(
            old_values,
            old_fitnesses,
            objective_sense=self.problem.senses[self._obj_index],
            ranking_method=self._ranking_method,
        )

        next_gen = self.step_count + 1
        freeze = (next_gen % self.freeze_interval != 0)

        # if freeze:
        #     if "d" in gradients and "M" in gradients:
        #         # "d" is shape (solution_length,)
        #         # "M" is shape (solution_length, solution_length)
        #         if self.freeze_params == "morphology":
        #             morph_slice = range(self.morph_start, self.morph_end)
        #             gradients["d"][morph_slice] = 0.0
        #             gradients["M"][morph_slice, :] = 0.0
        #             gradients["M"][:, morph_slice] = 0.0
        #         elif self.freeze_params == "controller":
        #             ctrl_slice = range(0, self.nn_params_size)
        #             gradients["d"][ctrl_slice] = 0.0
        #             gradients["M"][ctrl_slice, :] = 0.0
        #             gradients["M"][:, ctrl_slice] = 0.0

        self._update_distribution(gradients)

        self._population = SolutionBatch(
            self.problem,
            popsize=self._popsize,
            device=self._distribution.device,
            empty=True
        )
        self._distribution.sample(
            out=self._population.access_values(),
            generator=self.problem
        )

        # Overwrite the population's frozen portion
        newpop_vals = self._population.access_values()

        if freeze:
            for i in range(self._popsize):
                if self.freeze_params == "morphology":
                    newpop_vals[i, self.morph_start : self.morph_end] = self._stored_params[i]
                elif self.freeze_params == "controller":
                    newpop_vals[i, 0 : self.nn_params_size] = self._stored_params[i]
                else:
                    raise Exception("freeze_params option not supported")
        else:
            # Unfreeze => store the newly sampled portion
            for i in range(self._popsize):
                if self.freeze_params == "morphology":
                    self._stored_params[i] = newpop_vals[i, self.morph_start : self.morph_end]
                elif self.freeze_params == "controller":
                    self._stored_params[i] = newpop_vals[i, 0 : self.nn_params_size]
                else:
                    raise Exception("freeze_params option not supported")

        # Print debug info: morphological portion for individual 0
        print(f"=== Generation {next_gen} ===")
        slice_morph = newpop_vals[0, self.morph_start : self.morph_end]
        print(f"Ind 0 morph params: {slice_morph}")
        print("========================================")

        self.problem.evaluate(self._population)
