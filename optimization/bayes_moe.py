from collections import deque
import wrapt

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
from moe.optimal_learning.python.base_prior import BasePrior
from moe.optimal_learning.python.cpp_wrappers.optimization import (
    GradientDescentOptimizer as cGDOpt, GradientDescentParameters as cGDParams)
from moe.optimal_learning.python.base_prior import TophatPrior, NormalPrior

import logging

logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()

# Estimated from initial hyper-parameter optimization
DEFAULT_LENGTHSCALE_PRIOR = TophatPrior(-2, 5)
DEFAULT_CAMPL_PRIOR = NormalPrior(12.5, 1.6)
DEFAULT_PRIOR = DefaultPrior(n_dims=3 + 2, num_noise=1)
DEFAULT_PRIOR.tophat = DEFAULT_LENGTHSCALE_PRIOR
DEFAULT_PRIOR.ln_prior = DEFAULT_CAMPL_PRIOR

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


@wrapt.decorator
def _check_initialized(wrapped, instance, args, kwargs):
    if instance.gp_loglikelihood is None:
        logging.error("Model has not been initialized! "
                      "Use '.step() or .initialize_model()' "
                      "to initialize optimizer")
    else:
        return wrapped(*args, **kwargs)


class BayesianMOEOptimizer():
    def __init__(self,
                 objective_func,
                 samples_per_iteration,
                 bounds,
                 minimum_samples=10,
                 prior=None,
                 sgd_params=DEFAULT_SGD_PARAMS,
                 maximize=False,
                 epsilon=1e-3):
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
            maximize                    Boolean indicating whether the
                                        objective function is to be
                                        maximized instead of being minimized
            epsilon                     Standard deviation convergence
                                        threshold

        '''

        self.obj_func = objective_func
        self.maximize = maximize
        self.epsilon = epsilon

        self.dims = bounds.shape[0]
        self.best_point_history = []
        self.convergence_buffer = deque(maxlen=minimum_samples)

        # non-C++ wrapper needed since it has added functionality
        # at cost of speed
        moe_bounds = [ClosedInterval(mn, mx) for mn, mx in bounds]
        self.search_domain = TensorProductDomain(moe_bounds)
        self.c_search_domain = cTensorProductDomain(moe_bounds)

        # TODO: Noise modelling will be supported later
        if prior is None:
            logging.warning("Using default prior from Cornell MOE")
            logging.warning("Prior may be sub-optimal for problem " "domain!")
            self.prior = DefaultPrior(n_dims=self.dims + 2, num_noise=1)
        elif not isinstance(prior, BasePrior):
            raise ValueError("Prior must be of type BasePrior!")
        else:
            self.prior = prior

        self.sgd = cGDParams(**sgd_params)
        self.gp_loglikelihood = None

        self.num_samples = samples_per_iteration
        self.iteration = 0

    def _increment(self):
        self.iteration += 1

    @property
    def sign(self):
        return -1 if self.maximize else 1

    def __str__(self):
        return f'''
        Configuration:
        Samples/iteration: {self.num_samples}
        Minimum Samples: {self.convergence_buffer.maxlen}
        Epsilon: {self.epsilon}
        Bounds: {str(self.bounds)}
        Maximize: {self.maximize}

        State:
        Iteration: {self.iteration}
        Current Best: {self.current_best}
        Convergence: {self.convergence}
        '''

    @property
    def gp(self):
        '''
        Returns a single Gaussian process model
        from the current ensemble
        '''
        # TODO: Logging
        if self.gp_loglikelihood is None:
            logging.warning("Model has not been initialized "
                            "Use .initialize_model() or .step() to "
                            "initialize GP model")
            return
        return self.gp_loglikelihood.models[0]

    @property
    def current_best(self):
        '''
        Returns the current best input coordinate and value
        '''
        # TODO: Logging
        if self.gp_loglikelihood is None:
            logging.warning("Model has not been initialized "
                            "Use .initialize_model() or .step() to "
                            "initialize GP model")
            return
        history = self.gp.get_historical_data_copy()
        best_value = np.min(history._points_sampled_value)
        best_index = np.argmin(history._points_sampled_value)
        best_coord = history.points_sampled[best_index]
        return best_coord, best_value

    @property
    def converged(self):
        '''
        Evaluates whether Bayesian optimization has converged.
        Examines whether the standard deviation of the history of length
        `self.min_samples` is below `self.epsilon`

        If the minimum number of iterations have not been met, returns False
        '''
        # Minimum number of iterations have not been met
        if not len(self.convergence_buffer) == self.convergence_buffer.maxlen:
            return False

        if self.gp_loglikelihood is None:
            return False

        criterion = self._compute_convergence_criterion()
        logging.debug(f"Buffer standard deviation: {criterion}")
        return criterion < self.epsilon

    @_check_initialized
    def _compute_convergence_criterion(self):
        '''
        Compute the convergence criterion (sample standard deviation)
        of the past self.min_samples evaluations
        '''
        _, best = self.current_best
        deviation = np.linalg.norm(np.array(self.convergence_buffer) - best)
        return deviation

    def initialize_model(self):
        '''
        Initialize the GaussianProcessLogLikelihood model using
        an initial set of observations

        Returns:
            init_pts            Initial sampling points
            res                 Objective function evaluations of init_pts
        '''

        logging.debug(f"Initializing model with {self.num_samples} samples")
        init_pts = self.search_domain\
            .generate_uniform_random_points_in_domain(self.num_samples)

        res = self.evaluate_objective(init_pts)
        logging.debug(f"Initial samples: {init_pts}")

        history = HistoricalData(dim=self.dims, num_derivatives=0)
        history.append_sample_points(
            [SamplePoint(i, o, 0.0) for i, o in zip(init_pts, res)])

        self.gp_loglikelihood = GaussianProcessLogLikelihoodMCMC(
            historical_data=history,
            derivatives=[],
            prior=self.prior,
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=2**4,
            noisy=False)

        self.gp_loglikelihood.train()
        self._increment()
        return init_pts, res

    @_check_initialized
    def update_model(self, evidence):
        '''
        Updates the current ensemble of models with
        new data

        Arguments:
            evidence            New SamplePoint data
        '''
        self.gp_loglikelihood.add_sampled_points(evidence)
        self.gp_loglikelihood.train()
        return

    @_check_initialized
    def _update_history(self):
        '''
        Update the history of best points with the
        current best point and value
        '''
        best_coord, best_value = self.current_best
        self.best_point_history.append((best_coord, best_value))

    @_check_initialized
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

    def evaluate_objective(self, sampling_points):
        '''
        Evaluates the objective function at `sampling_points`

        The objective function is expected to return an iterable
        of the results with ordering matching the input sampling
        points
        '''

        res = self.obj_func(sampling_points)
        if not isinstance(res, np.ndarray):
            res = np.array(res)

        return self.sign * res

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
            sampling_points, res = self.initialize_model()
            qEI = None
            logging.debug("Model has not yet been built! "
                          "Initializing model..")
            self.initialize_model()
        else:
            sampling_points, qEI = self.propose_sampling_points()
            logging.debug(f"Sampling points: {str(sampling_points)}")
            logging.debug(f"q-Expected Improvement: {qEI}")
            res = self.evaluate_objective(sampling_points)
            evidence = [
                SamplePoint(c, v, 0.0) for c, v in zip(sampling_points, res)
            ]
            self.update_model(evidence)
        self._update_history()

        _, best = self.current_best
        self.convergence_buffer.append(best)
        self._increment()

        return sampling_points, res, qEI

    def iter(self):
        '''
        Returns an generator to perform
        end-to-end optimization
        '''
        while not self.converged:
            sampling_points, res, qEI = self.step()
            best_point, best_val = self.current_best
            yield {
                    "best_point": best_point,
                    "best_value": best_val,
                    "iteration": self.iteration,
                    "samples": sampling_points,
                    "result": res,
                    "qei": qEI,
                    "converged": self.converged
            }
        logging.debug(f"Current best is: {self.current_best}")
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
    # Set standard TMS bounds
    bounds = f.bounds
    bounds[2, :] = np.array([0, 180])

    return BayesianMOEOptimizer(objective_func=f.evaluate,
                                samples_per_iteration=num_samples,
                                bounds=bounds,
                                minimum_samples=minimum_samples,
                                prior=DEFAULT_PRIOR,
                                maximize=True)


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

    # lhc_iter=2e4 doesn't actually matter since we're using SGD
    optimizer = cGDOpt(search_domain, qEI, sgd_params, int(2e4))
    points_to_sample = meio(optimizer,
                            None,
                            num_samples,
                            use_gpu=False,
                            which_gpu=0,
                            max_num_threads=8)
    qEI.set_current_point(points_to_sample[0])

    return points_to_sample, qEI.compute_expected_improvement()
