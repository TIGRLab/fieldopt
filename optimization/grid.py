import numpy as np
from sklearn.utils.extmath import cartesian


class GridOptimizer():
    '''
    General GridOptimizer for an objective function
    '''
    def __init__(self,
                 objective_func,
                 batchsize,
                 sampling_density,
                 bounds,
                 maximize=True):
        '''
        Arguments:
            objective_func      Objective Function
            batchsize           Number of evaluations to perform in parallel
            sampling_density    [P] array indicating number of samples
                                for a dimension p
            bounds              [P x 2] Array where each row corresponds
                                to the (min, max) for a dimension p
        '''

        # Construct Grid
        dim_samples = [
            np.linspace(b[0], b[1], d)
            for b, d in zip(bounds, sampling_density)
        ]
        self.grid = cartesian(dim_samples)

        # Construct batches for evaluation
        divisions = np.arange(batchsize, self.grid.shape[0], batchsize)
        self.batches = np.split(self.grid, divisions)
        self.batchsize = batchsize

        self.obj_func = objective_func
        self.iteration = 0

        self.history = np.zeros((self.grid.shape[0], ), dtype=float)

        self.sign = -1 if maximize else 1

    def _increment(self):
        self.iteration += 1

    @property
    def completed(self):
        return self.iteration >= len(self.batches)

    @property
    def current_best(self):
        '''
        Returns:
            best_coord          Best parameter coordinate
            best_value          Current minimum of objective function
        '''

        # No history is recorded
        if self.iteration == 0:
            return

        ind = np.argmin(self.history)
        best_coord = self.grid[ind]
        best_value = self.history[ind]
        return best_coord, best_value

    def evaluate_objective(self, sampling_points):
        res = self.obj_func(sampling_points)
        if not isinstance(res, np.ndarray):
            res = np.array(res)

        return self.sign * res

    def step(self):
        '''
        Perform one iteration of grid optimization using
        up to self.batchsize evaluations

        Returns:
            sampling_points     Points sampled on iteration
            res                 Resultant values
        '''

        if self.completed:
            raise StopIteration

        sampling_points = self.batches[self.iteration]
        res = self.evaluate_objective(sampling_points)

        block_start = self.iteration * self.batchsize
        block_end = block_start + len(res)
        self.history[block_start:block_end] = res

        self._increment()
        return sampling_points, res

    def into_iter(self):
        '''
        Returns an generator which can be run to
        perform end-to-end optimization
        '''

        while not self.completed:
            sampling_points, res = self.step()
            best_point, best_val = self.current_best
            yield {
                "best_point": best_point,
                "best_value": best_val,
                "iteration": self.iteration,
                "samples": sampling_points,
                "result": res
            }


def get_default_tms_optimizer(f, locdim, rotdim):
    '''
    Construct GridOptimizer for FEM evaluation. The number
    of evaluations per iteration will be set to f.cpus//2 - 1
    so that SimNIBS doesn't hang when cgroup restrictions are
    placed on CPU resources

    Arguments:
        f           FieldFunc objective function
        locdim      Number of spatial positions to evaluate along the x/y
                    dimension. The total number of spatial positions sampled
                    will be locdim * locdim
        rotdim      Number of orientations to sample. The range of orientations
                    sampled will be constrained to [0, 180]

    '''

    sampling = (locdim, locdim, rotdim)
    batchsize = f.cpus // 2 - 1
    bounds = f.bounds
    bounds[2, :] = np.array([0, 180])
    return GridOptimizer(f.evaluate, batchsize, sampling, bounds)
