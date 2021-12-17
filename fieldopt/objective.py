# coding: utf-8

import logging
import time
import multiprocessing as mp
import copy

import numpy as np
import mkl
import scipy.sparse as sparse

from simnibs import cond
from simnibs.simulation.fem import FEMSystem, grad_matrix, calc_fields
from simnibs.msh import mesh_io
import simnibs.simulation.coil_numpy as coil_lib
from fieldopt.solver_wrapper import get_solver

logger = logging.getLogger(__name__)

FIELD_ENTITY = (3, 2)
PIAL_ENTITY = (2, 1002)
FIELDS = ['e']


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
    def __init__(self,
                 head_model,
                 sampling_domain,
                 coil,
                 tet_weights=None,
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

        if tet_weights:
            self.tw = tet_weights[np.where(tet_weights)]
        else:
            logger.warning("No weights provided for score function"
                           " setting to 0s!")
            self.tw = np.zeros_like(head_model.get_tet_ids(2)[0])

        # Control for coil file-type and SimNIBS changing convention
        self.normflip = coil.endswith('.nii.gz')

        roi = head_model.get_tet_ids(2)[0][np.where(tet_weights)]

        self.simulator = _Simulator(head_model, solver, didt, coil, roi,
                                    nworkers, nthreads)

        self.volumes = self.mesh.elements_volumes_and_areas().value[roi]

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.model.mesh)
        return ''

    @property
    def mesh(self):
        return self.model.mesh

    def place_coils(self, input_list):
        '''
        For an input list of triplets (x,y,theta), place coils
        on the head model according to the defined Domain `self.domain`

        Arguments:
            input_list              List of (x,y,t) triplets
        '''

        logger.info("Transforming inputs...")
        return [
            self.domain.place_coil(self.model, *i, self.normflip)
            for i in input_list
        ]

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

        matsimnibs = self.place_coils(input_list)

        logger.info('Running simulations...')
        E = self.simulator.run_simulation(self.mesh, matsimnibs)
        logger.info('Successfully completed simulations!')
        scores = (np.linalg.norm(E, axis=1) * self.tw[:, None] *
                  self.volumes[:, None]).sum(axis=0)
        return scores

    def visualize_evaluate(self,
                           x=None,
                           y=None,
                           theta=None,
                           matsimnibs=None,
                           out_sim=None,
                           out_geo=None,
                           fields=FIELDS):
        '''
        Generate a simulation visualization for a given (x,y,theta) on the
        configured `sampling_domain`.

        Arguments:
            x                       Sampling domain x
            y                       Sampling domain y
            theta                   Sampling domain rotation angle
            matsimnibs              Matsimnibs matrix to use directly
            out_sim                 Output path for msh file
            out_geo                 Output path for coil position file
            fields                  Fields to include in visualization
        '''

        if all([x, y, theta]):
            logger.info("Using domain input coordinates")
            matsimnibs = self.place_coils([[x, y, theta]])[0]
        elif matsimnibs is not None:
            logger.info("Using provided matsimnibs matrix")
        else:
            raise ValueError("Missing x,y,theta coordinates or matsimnibs!")

        res = self.simulator.run_full_simulation(self.mesh,
                                                 matsimnibs,
                                                 fields=fields,
                                                 fn_geo=out_geo)

        if np.any(np.where(self.tw)[0]):
            logger.info("Appending weightfunction to output")
            wf_field = np.zeros_like(res.elmdata[0].value)
            wf_field[self.simulator.roi] = self.tw
            res.add_element_field(wf_field, 'weightfunction')
            return
        else:
            logger.warning("Weightfunction was not set or is all"
                           " zeros! Not displaying in final result")

        mesh_io.write_msh(res, out_sim)
        logger.info(f"Wrote msh into {out_sim}")

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
        return np.linalg.norm(self.model.intercept(p0, p1, PIAL_ENTITY))


class _Simulator:
    '''
    Class to manage simulation environment
    '''
    def __init__(self,
                 model,
                 solver,
                 didt,
                 coil,
                 roi=None,
                 num_workers=None,
                 nthreads=None):

        self.FEMSystem = None
        self.solver = None
        self.cond = None
        self.didt = didt
        self.coil = coil
        self.num_workers = num_workers if num_workers else 1
        if nthreads:
            self.set_num_threads(nthreads)

        logger.info('Storing standard conductivity values...')
        condlist = [c.value for c in cond.standard_cond()]
        conductivities = cond.cond2elmdata(model.mesh, condlist)
        self.cond = conductivities
        logger.info('Successfully stored conductivity values...')

        logger.info("Initializing FEM problem")
        self.FEMSystem = FEMSystem.tms(model.mesh, conductivities)
        logger.info("Completed FEM initialization")

        logger.info('Computing mesh gradient')
        D = grad_matrix(model.mesh, split=True)
        if roi is None:
            roi = model.get_tet_ids(FIELD_ENTITY[1])
        self.D = [d.tocsc()[roi] for d in D]
        self.roi = roi

        logger.info("Computing A matrix of AX=B")
        A = sparse.csc_matrix(self.FEMSystem.A)
        A.sort_indices()
        dof_map = copy.deepcopy(self.FEMSystem.dof_map)
        if self.FEMSystem.dirichlet is not None:
            A, _ = self.FEMSystem.dirichlet.apply_to_matrix(A, dof_map)
        logger.info("Preparing solver...")
        self.solver = get_solver(solver, A)
        logger.info("Completed initialization successfully!")

    def __getstate__(self):
        state = {}
        state["didt"] = self.didt
        state["coil"] = self.coil
        state["FEMSystem"] = self.FEMSystem
        state["roi"] = self.roi
        state["D"] = self.D
        return state

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

    def prepare_tms_matrix(self, mesh, m, fn_geo=None):
        '''
        Construct right hand side of the TMS linear
        problem

        Arguments;
            mesh                    simnibs.msh.mesh_io.Msh object
            m                       Matsimnibs matrix

        Returns:
            b                       RHS of simulation
            dadt                    dA/dt for a given TMS coil position
        '''

        dadt = coil_lib.set_up_tms(mesh,
                                   self.coil,
                                   m,
                                   self.didt,
                                   fn_geo=fn_geo)
        b = self.FEMSystem.assemble_tms_rhs(dadt)
        return b, dadt

    def calc_E(self, V, dAdt):

        E = np.stack([-d.dot(V) for d in self.D], axis=1) * 1e3
        E -= dAdt
        return E

    def solve(self, B):
        '''
        Solve FEM

        B           Right hand-side of TMS FEM equation
        '''

        B, dof_map = self.FEMSystem.dirichlet.apply_to_rhs(
            self.FEMSystem.A, B, self.FEMSystem.dof_map)

        logger.info("Solving FEM")
        X = self.solver.solve(B)
        logger.info("Solved")

        X, dof_map = self.FEMSystem.dirichlet.apply_to_solution(X, dof_map)
        _, X = dof_map.order_like(self.FEMSystem.dof_map, array=X)

        return X

    def run_simulation(self, mesh, matsimnibs):

        if not isinstance(matsimnibs, list):
            matsimnibs = [matsimnibs]

        logger.info('Constructing right-hand side of FEM AX=B...')
        start = time.time()

        args = [(mesh, m) for m in matsimnibs]
        with mp.Pool(processes=self.num_workers) as pool:
            res = pool.starmap(self.prepare_tms_matrix, args)
            end = time.time()
            logger.info(f"Took {end-start:.2f}")

        # Each column vector is a TMS coil position
        bs = []
        dadts = []
        for b, dadt in res:
            bs.append(b)
            dadts.append(dadt.value[self.roi])

        # [Node x problems]
        B = np.vstack(bs).T
        X = self.solve(B)

        logger.info("Computing E")
        # [Nodes x Directions x Problems]
        DADT = np.stack(dadts, axis=2)
        E = self.calc_E(X, DADT)
        logger.info("Done")

        return E

    def run_full_simulation(self,
                            mesh,
                            matsimnibs,
                            fields=FIELDS,
                            fn_geo=None):

        logger.info('Constructing right-hand side of FEM AX=B...')
        b, dadt = self.prepare_tms_matrix(mesh, matsimnibs, fn_geo=fn_geo)
        x = self.solve(b)

        # Calc fields
        logger.info("Computing output post-processing")
        x = mesh_io.NodeData(x.squeeze(), name='v', mesh=mesh)
        x.mesh = mesh
        res = calc_fields(x, fields, cond=self.cond, dadt=dadt)
        logger.info("Done")

        # Return resulting mesh object
        return res
