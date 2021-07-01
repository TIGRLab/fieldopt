from collections import deque

import numpy as np

from moe.optimal_learning.python.cpp_wrappers.domain import (
    TensorProductDomain as cTensorProductDomain)
from moe.optimal_learning.python.python_version.domain import (
    TensorProductDomain)
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import (
    ExpectedImprovement)
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import (
    multistart_expected_improvement_optimization as meio)
from moe.optimal_learning.python.data_containers import (HistoricalData,
                                                         SamplePoint)
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import (
    GaussianProcessLogLikelihoodMCMC)
from moe.optimal_learning.python.default_priors import DefaultPrior
from moe.optimal_learning.python.cpp_wrappers.optimization import (
    GradientDescentOptimizer as cGDOpt, GradientDescentParameters as cGDParams)
from moe.optimal_learning.python.base_prior import TophatPrior, NormalPrior

# Estimated from initial hyper-parameter optimization
DEFAULT_LENGTHSCALE_PRIOR = TophatPrior(-2, 5)
DEFAULT_CAMPL_PRIOR = NormalPrior(12.5, 1.6)
DEFAULT_PRIOR = DefaultPrior(n_dims=3 + 2, num_noise=1)
DEFAULT_PRIOR.tophat = DEFAULT_CAMPL_PRIOR
DEFAULT_PRIOR.ln_prior = DEFAULT_LENGTHSCALE_PRIOR

# Default SGD
DEFAULT_SGD_PARAMS = {
    "num_multistarts": 200,
    "max_num_steps": 50,
    "max_num_restarts": 5,
    "num_steps_averaged": 4,
    "gamma": 0.7,
    "pre_mult": 1.0,
    "max_relative_change": 0.5,
    "tolerance": 1.0e-10
}


class BayesianMOEOptimizer():
    def __init__(self,
                 objective_func,
                 samples_per_iteration,
                 bounds,
                 minimum_samples=10,
                 prior=DefaultPrior,
                 sgd_params=DEFAULT_SGD_PARAMS):
        '''
        Initialize default parameters for running a Bayesian optimization
        algorithm on an objective function with parameters.

        Initializes a squared exponential covariance prior using a
        - tophat prior on the lengthscale
        - lognormal prior on the covariance amplitude

        Arguments:
            objective_func              Objective function
            samples_per_iteration       Number of samples to propose
                                        per iteration
            bounds                      [P x 2] array where each row
                                        corresponds to the (min, max)
                                        of feature P
            minimum_samples             Minimum number of samples to collect
                                        before performing convergence checks
            prior                       Cornell MOE BasePrior subclass
                                        to use for lengthscale and
                                        covariance amplitude
            sgd_params                  Stochastic Gradient Descent parameters
                                        (see: cornell MOE's
                                        GradientDescentParameters)

        '''

        self.obj_func = objective_func

        self.dims = bounds.shape[0]
        self.best_point_history = []
        self.convergence_buffer = deque(maxlen=minimum_samples)

        # non-C++ wrapper needed since it has added functionality
        # at cost of speed
        moe_bounds = [ClosedInterval(mn, mx) for mn, mx in bounds]
        self.search_domain = TensorProductDomain(moe_bounds)
        self.c_search_domain = cTensorProductDomain(moe_bounds)

        # TODO: Noise modelling will be supported later
        self.prior = prior(n_dims=self.dims + 2, num_noise=1)
        self.sgd = cGDParams(**sgd_params)
        self.gp_loglikelihood = None

        self.num_samples = samples_per_iteration
        self.iteration = 0

    def _increment(self):
        self.iteration += 1

    @property
    def gp(self):
        '''
        Returns a single Gaussian process model
        from the current ensemble
        '''
        return self.gp_likelihood[0]

    @property
    def current_best(self):
        '''
        Returns the current best input coordinate and value
        '''
        history = self.gp.get_historical_data_copy()
        best_value = np.min(history._points_sampled_value)
        best_index = np.argmin(history._points_sampled_value)
        best_coord = history.pointed_sampled[best_index]
        return best_coord, best_value

    def initialize_model(self):
        '''
        Initialize the GaussianProcessLogLikelihood model using
        an initial set of observations
        '''

        init_pts = self.search_domain\
            .generate_uniform_random_points_in_domain(self.num_samples)
        observations = self.obj_func(init_pts)

        history = HistoricalData(dim=self.dims, num_derivatives=0)
        history.append_sample_points(
            [SamplePoint(i, o, 0.0) for i, o in zip(init_pts, observations)])

        self.gp_likelihood = GaussianProcessLogLikelihoodMCMC(
            historical_data=history,
            derivatives=[],
            prior=self.prior,
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=2**4,
            noisy=False)

        self.increment()

    def update_model(self, evidence):
        '''
        Updates the current ensemble of models with
        new data

        Arguments:
            evidence            New SamplePoint data
        '''
        self.gp_likelihood.add_sampled_points(evidence)
        self.gp_likelihood.train()
        return

    def _update_history(self):
        '''
        Update the history of best points with the
        current best point and value
        '''
        best_coord, best_value = self.best_coord
        self.best_point_history.append((best_coord, best_value))

    def propose_sampling_points(self):
        '''
        Performs stochastic gradient descent to optimize qEI function
        returning a list of optimal candidate points for the current
        set of ensemble models

        Returns:
            samples         Set of optimal samples to evaluate
            ei              Expected improvement
        '''
        samples, ei = _gen_sample_from_qei(self.gp, self.c_search_domain,
                                           self.sgd, self.num_samples)
        return samples, ei

    def has_converged(self):
        '''
        Evaluates whether Bayesian optimization has converged.
        Examines whether the standard deviation of the history of length
        `self.min_samples` is below `self.tolerance`
        '''
        best = np.min(self.gp._points_sampled_value)
        deviation = sum([abs(x - best) for x in self.convergence_buffer])
        return deviation < self.tolerance

    def evaluate_objective(self, sampling_points):
        '''
        Evaluates the objective function at `sampling_points`

        The objective function is expected to return an iterable
        of the results with ordering matching the input sampling
        points
        '''

        res = self.obj_func(sampling_points)
        return res

    def step(self):
        '''
        Performs one iteration of Bayesian optimization:
            1. get sampling points via maximizing qEI
            2. Evaluate objective function at proposed sampling points
            3. Update ensemble of Gaussian process models
            4. Store current best in the history of best points
            5. Increment iteration counter
        '''
        if self.iteration == 0:
            self.initialize_model()
        else:
            sampling_points = self.propose_sampling_points()
            res = self.evaluate_objective(sampling_points)
            evidence = [
                SamplePoint(c, v, 0.0) for c, v in zip(sampling_points, res)
            ]
            self.update_model(evidence)
        self._update_history()
        self._increment()
        return


def get_default_tms_optimizer(f, num_samples, minimum_samples=10):
    '''
    Construct BayesianMOEOptimizer using pre-configured
    prior hyperparameters

    Arguments:
        f                   FieldFunc objective function
        num_samples         Number of samples to evaluate in parallel
        minimum_samples     Minimum number of samples to evaluate before
                            performing convergence checks
    Returns:
        Configured BayesianMOEOptimizer instance with the following priors:
        - Squared exponential covariance function:
            - Length scale with a TopHat(-2, 5)
            - Log-normal covariance amplitude Ln(Normal(12.5, 1.6))
    '''

    # MOE minimizes the objective function
    def obj(f):
        return -f.evaluate

    # Set standard TMS bounds
    bounds = np.r_[f.bounds, np.array([0, 180])]

    return BayesianMOEOptimizer(objective_func=obj,
                                samples_per_iteration=num_samples,
                                bounds=bounds,
                                minimum_samples=minimum_samples,
                                prior=DEFAULT_PRIOR)


def _gen_sample_from_qei(gp,
                         search_domain,
                         sgd_params,
                         num_samples,
                         num_mc=1e4):
    '''
    Perform multistart stochastic gradient descent (MEIO)
    on the q-EI of a gaussian process model with
    prior models on its hyperparameters

    Arguments:
        gp              Gaussian process model
        search_domain   Input domain of gaussian process model
        sgd_params      Stochastic gradient descent parameters
        num_samples     Number of samples to maximize over
        num_mc          Number of monte carlo sampling iterations to
                        perform to compute integral

    Returns:
        points_to_sample    Optimal samples to evaluate
        qEI                 q-Expected Improvement of `points_to_sample`
    '''

    qEI = ExpectedImprovement(gaussian_process=gp,
                              num_mc_iterations=int(num_mc))
    optimizer = cGDOpt(search_domain, qEI, sgd_params)
    points_to_sample = meio(optimizer,
                            None,
                            num_samples,
                            use_gpu=False,
                            which_gpu=0,
                            max_num_threads=8)
    qEI.set_current_point(points_to_sample[0])

    return points_to_sample, qEI.compute_expected_improvement()
