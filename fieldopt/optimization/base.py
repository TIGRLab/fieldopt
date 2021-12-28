"""
Base classes for Fieldopt optimizers
"""

from abc import ABC, abstractmethod
import numpy as np


class IterableOptimizer(ABC):
    '''
    ABC for implementing basic iterable optimizers
    '''
    @abstractmethod
    def __init__(self, objective_func, maximize=False):
        self.obj_func = objective_func
        self.maximize = maximize
        self.iteration = 0

    @abstractmethod
    def iter():
        pass

    @abstractmethod
    def get_history():
        pass

    @abstractmethod
    def current_best():
        pass

    @property
    def sign(self):
        '''
        Sign of objective function

        Returns:
            sign (float): -1 if maximization problem, 1 if minimization
        '''
        return -1 if self.maximize else 1

    def _increment(self):
        self.iteration += 1

    def optimize(self, print_status=False):
        '''
        End-to-end optimization of an objective function

        Returns:
            history list(dict): Optimization historical data
        '''
        history = [res for res in self.iter(print_status)]
        return history

    def evaluate_objective(self, sampling_points):
        '''
        Evaluates the objective function at `sampling_points`

        The objective function is expected to return an iterable
        of the results with ordering matching the input sampling
        points

        Arguments:
            sampling_points (ndarray): (N,P) points to sample
                objective function at

        Returns:
            obj (ndarray): (N,) Objective vector
        '''

        res = self.obj_func(sampling_points)
        if not isinstance(res, np.ndarray):
            res = np.array(res)

        return self.sign * res
