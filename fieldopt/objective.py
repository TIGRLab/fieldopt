# coding: utf-8

import logging
import time
import multiprocessing as mp
import copy

import numpy as np
import mkl
import scipy.sparse as sparse

from simnibs import cond
from simnibs.msh import mesh_io
from simnibs.simulation.fem import FEMSystem, calc_fields
import simnibs.simulation.coil_numpy as coil_lib
from fieldopt.solver_wrapper import get_solver

logger = logging.getLogger(__name__)


class FieldFunc():
    '''
    This class provides an interface in which the details related
    to simulation and score extraction is abstracted away for
    any general Optimization package

    Layout
    Properties:
        1. Mesh file to perform simulation on
        2. Store details surrounding the input -->
            surface transformations (use file paths?)

    '''

    FIELD_ENTITY = (3, 2)
    PIAL_ENTITY = (2, 1002)
    FIELDS = ['e']

    def __init__(self,
                 head_model,
                 sampling_domain,
                 tet_weights,
                 coil,
                 didt=1e6,
                 nworkers=1,
                 nthreads=None,
                 solver='pardiso'):
        '''
        Standard constructor
        Arguments:
            mesh                        simnibs.Msh.mesh_io.Msh object
            sampling_domain             fieldopt.geometry.sampler Domain class
            tet_weights                 Weighting scores for each tetrahedron
                                        (1D array ordered by node ID)
            coil                        TMS coil file (either dA/dt volume or
                                        coil geometry)
            didt                        Intensity of stimulation
            nworkers                    Number of workers to spawn
                                        in order to set up simulations
                                        (compute B)
            nthreads                    Maximum number of physical cores
                                        for solver to use [Default: use all]
            solver                      Solver to use to compute FEM solution
                                        [Default: pardiso]
        '''

        self.model = head_model
        self.domain = sampling_domain
        self.tw = tet_weights
        self.coil = coil
        self.didt = didt

        # Control for coil file-type and SimNIBS changing convention
        self.normflip = self.coil.endswith('.nii.gz')

        self.FEMSystem = None
        self.initialize(solver)
        self.num_workers = nworkers

        if nthreads:
            self.set_num_threads(nthreads)

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.model.mesh)
        print('Coil:', self.coil)
        return ''

    def __getstate__(self):
        state = {}
        state["didt"] = self.didt
        state["model"] = self.model
        state["coil"] = self.coil
        state["FEMSystem"] = self.FEMSystem
        state["cond"] = self.cond
        state["tw"] = self.tw
        return state

    @property
    def mesh(self):
        return self.model.mesh

    def set_num_threads(self, mkl_threads):

        if mkl_threads > mp.cpu_count() // 2:
            logger.warning("Maximum number of cores requested "
                           " exceeds number of physical cores!")
            logger.warning(f"Physical Cores: {mp.cpu_count()}")
            logger.warning(f"Maximum usable cores: {mkl_threads}")
            logger.warning("Using all cores")
        else:
            logger.info("Setting maximum number of usable physical cores"
                        f"to {mkl_threads}")
            mkl.set_dynamic(0)
            mkl.set_num_threads(mkl_threads)

    def initialize(self, solver):
        '''
        Initialize objective function to allow for objective
        function evaluations.

        Caches:
            - Conductivity list
            - FEMSystem

        Opens multiprocessing pool to have jobs submitted
        '''

        logger.info('Storing standard conductivity values...')
        condlist = [c.value for c in cond.standard_cond()]
        self.cond = cond.cond2elmdata(self.model.mesh, condlist)
        logger.info('Successfully stored conductivity values...')

        logger.info("Initializing FEM problem")
        self.FEMSystem = FEMSystem.tms(self.model.mesh, self.cond)
        logger.info("Completed FEM initialization")

        logger.info("Computing A matrix of AX=B")
        A = sparse.csc_matrix(self.FEMSystem.A)
        A.sort_indices()
        dof_map = copy.deepcopy(self.FEMSystem.dof_map)
        if self.FEMSystem.dirichlet is not None:
            A, _ = self.FEMSystem.dirichlet.apply_to_matrix(A, dof_map)
        logger.info("Preparing solver...")
        self.solver = get_solver(solver, A)
        logger.info("Completed initialization successfully!")
        return

    def _prepare_tms_matrix(self, m):
        '''
        Construct right hand side of the TMS linear
        problem
        '''

        # Compute dA/dt for coil posiion on mesh
        dadt = coil_lib.set_up_tms(self.mesh, self.coil, m, self.didt)
        dof_map = copy.deepcopy(self.FEMSystem.dof_map)

        # Assemble b
        b = self.FEMSystem.assemble_tms_rhs(dadt)
        b, dof_map = self.FEMSystem.dirichlet.apply_to_rhs(
            self.FEMSystem.A, b, dof_map)
        return b, dof_map

    def _run_simulation(self, matsimnibs, out):

        if not isinstance(matsimnibs, list):
            matsimnibs = [matsimnibs]

        if out:
            out_geos = [f"{out}_coil_{i}.geo" for i in range(len(matsimnibs))]
            out_sims = [
                f"{out}_fields_{i}.msh" for i in range(len(matsimnibs))
            ]
        else:
            out_geos = [None] * len(matsimnibs)
            out_sims = [None] * len(matsimnibs)

        # Only this part needs to be multiproc'd
        logger.info('Constructing right-hand side of FEM AX=B...')
        start = time.time()
        with mp.Pool(processes=self.num_workers) as pool:
            res = pool.map(self._prepare_tms_matrix, matsimnibs)
            end = time.time()
            logger.info(f"Took {end-start:.2f}")

            # Each column vector is a TMS coil position
            bs = []
            dofs = []
            for b, d in res:
                bs.append(b)
                dofs.append(d)

            B = np.stack(bs, axis=1)
            X = self.solver.solve(B).squeeze()

            scores = []
            for i in range(X.shape[1]):
                scores.append(
                    pool.apply_async(self._compute_score, (
                        X[:, i],
                        dofs[i],
                    )))
            [s.get() for s in scores]
        return scores

    def _compute_score(self, v, dof):
        if self.FEMSystem.dirichlet is not None:
            v, dof_map = self.FEMSystem.dirichlet.apply_to_solution(v, dof)
        V = mesh_io.NodeData(v.squeeze(), name="v", mesh=self.mesh)
        V.mesh = self.mesh
        out = calc_fields(V, self.FIELDS, cond=self.cond)
        return self._calculate_score(out)

    def _calculate_score(self, sim_msh):
        '''
        Given a simulation output file, compute the score

        Volumes are integrated into the score function to deal with
        score inflation due to a larger number of elements near
        more complex geometry.
        '''

        tet_ids = self.model.get_tet_ids(self.FIELD_ENTITY[1])
        normE = sim_msh.elmdata[0].value[tet_ids]

        neg_ind = np.where(normE < 0)
        normE[neg_ind] = 0

        vols = self.mesh.elements_volumes_and_areas()[tet_ids]

        scores = self.tw * normE * vols
        return scores.sum()

    def evaluate(self, input_list, out_basename=None):
        '''
        Given a quadratic surface input (x,y) and rotational
        interpolation angle (theta) compute the resulting field score
        over a region of interest
        Arguments:
            [(x,y,theta),...]           A iterable of iterable (x,y,theta)

        Returns:
            scores                      An array of scores in order of inputs
        '''

        logger.info('Transforming inputs...')
        matsimnibs = [
            self.domain.place_coil(self.model, x, y, t, self.normflip)
            for x, y, t in input_list
        ]

        logger.info('Running simulations...')
        scores = self._run_simulation(matsimnibs, out_basename)
        logger.info('Successfully completed simulations!')

        return scores

    def get_coil2cortex_distance(self, input_coord):
        '''
        Given an input sampling coordinate on the parameteric
        mesh calculate the distance from the coil to target triangle
        on the cortical surface mesh.

        This function wraps:
        `simnibs.Msh.intercept_ray`

        The proposed ray is drawn from the coil centre and is drawn
        with large magnitude in the direction of the coil normal

        Arguments:
            input_coord : a 1 dimensional 3-iterable containing
                x, y, and rotation

        Returns:
            d : The distance between the proposed coil location
                and the target ROI on the cortex
        '''

        # Compute the coil affine matrix
        coil_affine = self.domain.place_coil(self.model, *input_coord,
                                             self.normflip)
        n = coil_affine[:3, 2]
        p0 = coil_affine[:3, 3]
        p1 = p0 + (n * 200)
        return np.linalg.norm(self.model.intercept(p0, p1, self.PIAL_ENTITY))
