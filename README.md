In Mujoco replace the following line on 593
myenv/lib/python3.10/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py

From this:
bottomleft, "Solver iterations", str(self.data.solver_iter + 1)

To this:
bottomleft, "Solver iterations", str(self.data.solver_niter[0] + 1)