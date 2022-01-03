# coding: utf-8
'''
Lightweight wrapper to provide equivalent interface between solvers
used for FEM
'''

import time
import atexit
import logging
from pypardiso import PyPardisoSolver
from simnibs.cython_code import petsc_solver
from simnibs.simulation.fem import DEFAULT_SOLVER_OPTIONS

logger = logging.getLogger(__name__)


class Pardiso:
    """
    PARDISO solver wrapper
    """
    def __init__(self, A):
        logger.info("Factorizing A for PARDISO solver")
        start = time.time()
        self.solver = PyPardisoSolver()
        self.solver.factorize(A)
        end = time.time()
        logger.info(f"Factorized A in {end - start:.2f} seconds")
        self._A = A
        atexit.register(self.solver.free_memory)

    def solve(self, B):
        """
        Solve linear equation :math:`Ax=B`

        Arguments:
            B (ndarray): (N,P) array of :math:`P` problems for
                :math:`N` elements

        Returns:
            X (ndarray): (N,P) Solutions matrix
        """
        return self.solver.solve(self._A, B)


class PETSc:
    """
    PETSc solver wrapper
    """
    def __init__(self, A, solver_opt=DEFAULT_SOLVER_OPTIONS):
        logger.info("Using PetSC solver")
        self.A = A
        self.solver_opt = solver_opt
        logger.info("Initialized PetSC!")
        petsc_solver.petsc_initialize()
        atexit.register(petsc_solver.petsc_finalize)

    def solve(self, B):
        """
        Solve linear equation :math:`Ax=B`

        Arguments:
            B (ndarray): (N,P) array of :math:`P` problems for
                :math:`N` elements

        Returns:
            X (ndarray): (N,P) Solutions matrix
        """
        petsc_solver.petsc_solve(self.solver_opt, self.A, B)


def get_solver(solver, A):
    """
    Get solver using string names

    Arguments:
        solver (str): Name of solver to initialize
        A (ndarray): (N, N) Left-hand matrix of :math:`AX=B` to solve

    Returns:
        Solver interface providing a `solver.solve(B)` method

    Note:
        Valid solvers are: ["pardiso", "petsc"]

    Raises:
        KeyError: If `solver` does not match a valid solver
    """
    solvers = {"pardiso": Pardiso, "petsc": PETSc}

    if solvers != "petsc":
        # Initializing simnibs.fem forces petsc initialization which
        # causes annoying false error messages.
        # If we're not using petsc shut it down
        petsc_solver.petsc_finalize()
        atexit.unregister(petsc_solver.petsc_finalize)

    try:
        return solvers[solver](A)
    except KeyError:
        logging.error(f"Solver {solver} not available!")
        available_solvers = "\n".join(solvers.keys())
        logger.error(f"Available solvers:\n {available_solvers}")
        raise
