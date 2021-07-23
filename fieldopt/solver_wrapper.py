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

# Match SimNIBS module-level setup
petsc_solver.petsc_initialize()
atexit.register(petsc_solver.petsc_finalize)


class Pardiso:
    def __init__(self, A):
        logger.info("Factorizing A for PARDISO solver")
        start = time.time()
        self.solver = PyPardisoSolver()
        self.solver.factorize(A)
        end = time.time()
        logger.info(f"Factorized A in {end - start:.2f} seconds")
        self._A = A

    def solve(self, B):
        return self.solver.solve(self._A, B)

class PETSc:
    def __init__(self, A, solver_opt=DEFAULT_SOLVER_OPTIONS):
        logger.info("Using PetSC solver")
        self.solver = petsc_solver.Solver(solver_opt, A)
        logger.info("Initialized PetSC!")


    def solve(self, B):
        self.solver.solve(B)

def get_solver(solver, A):
    solvers = {
            "pardiso": Pardiso,
            "petsc": PETSc
    }

    try:
        return solvers[solver](A)
    except KeyError:
        logging.error(f"Solver {solver} not available!")
        available_solvers = "\n".join(solvers.keys())
        logger.error(f"Available solvers:\n {available_solvers}")
        raise
