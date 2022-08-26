# coding: utf-8
'''
Objective function for evaluating TMS field simulations
'''

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
FIELDS = ['E']


class FieldFunc():
    '''
    TMS simulation objective function
    '''
    def __init__(self,
                 head_model,
                 sampling_domain,
                 coil,
                 tet_weights=None,
                 didt=1e6,
                 direction=None,
                 nworkers=1,
                 nthreads=None,
                 solver='pardiso'):
        '''
        Arguments:
            mesh: simnibs.Msh.mesh_io.Msh object
            sampling_domain: fieldopt.geometry.sampler Domain class
            tet_weights (ndarray): (T,) Weighting scores for each tetrahedron
            coil (str): TMS coil file (either dA/dt volume or
                coil geometry)
            didt (float): Intensity of stimulation
            direction (Optional[ndarray]): Project E-field to provided
                [3,] direction then calculate magnitude. Will use
                un-projected magnitude if not provided.
            nworkers (int): Number of workers to spawn for simulations
            nthreads (int): Maximum number of physical cores
                for solver to use [Default: use all]
            solver (str): Solver to use to compute FEM solution
                [Default: pardiso]. See fieldopt.solvers for list of
                available solvers
        '''

        self.model = head_model
        self.domain = sampling_domain

        if tet_weights is not None:
            self.tw = tet_weights[np.where(tet_weights)]
            roi = head_model.get_tet_ids(2)[0][np.where(tet_weights)]
        else:
            logger.warning("No weights provided for score function"
                           " setting to 0s!")
            self.tw = np.zeros_like(head_model.get_tet_ids(2)[0])
            roi = None

        self.simulator = _Simulator(head_model, solver, didt, coil, roi,
                                    nworkers, nthreads)

        self.volumes = self.mesh.elements_volumes_and_areas().value[roi]

        if direction is None:
            self.direction = direction
        else:
            self.direction = np.zeros((3, 1), dtype=float)
            self.direction[:, 0] = direction / np.linalg.norm(direction)

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.model.mesh)
        return ''

    @property
    def mesh(self):
        '''
        Get underlying simnibs.msh.Msh object
        '''
        return self.model.mesh

    def place_coils(self, input_list):
        '''
        Place multiple coil locations on parameteric surface defined
        by objective funtion domain

        Arguments:
            input_list (List[tuple(float,float,float]): (N,3) iterable
                containing (x,y,theta) entries

        Returns:
            List of SimNIBS matsimnibs matrices of length N
        '''

        logger.info("Transforming inputs...")
        return [self.domain.place_coil(self.model, *i) for i in input_list]

    def _compute_efield_magnitude(self, E, zero_negatives=False):
        '''
        Compute the signed magnitude of a (T,3) or (T,3,M)
        E field matrix where:
            - T: Number of tetrahedrons
            - 3: X,Y,Z (RAS) E-field vector
            - M: Number of TMS problems

        Arguments:
            E (ndarray): A (T,3) or (T,3,M) E-field matrix
            zero_negatives (bool): Set negative values to zero

        Returns:
            norms (ndarray): (T,M) norms array. If a 2D E-field
                array was provided, then a (T,1) array is produced
        '''

        if self.direction is None:
            norms = np.linalg.norm(E, axis=1)
        else:
            if len(E.shape) == 2:
                norms = E @ self.direction
            elif len(E.shape) == 3:
                norms = np.einsum('ijk,jl->ilk', E, self.direction)
                norms = np.squeeze(norms, axis=2)

        if zero_negatives:
            norms = norms.clip(min=0)
        return norms

    def evaluate(self, input_list, out_basename=None):
        '''
        Evaluate the TMS objective function on a list of input
        coordinates

        Arguments:
            input_list (List[tuple(float,float,float]): (N,3) iterable
                containing (x,y,theta) entries

        Returns:
            (N,) array of total E-field magnitudes over ROI for each
                input position
        '''

        matsimnibs = self.place_coils(input_list)

        logger.info('Running simulations...')
        E = self.simulator.run_simulation(self.mesh, matsimnibs)
        logger.info('Successfully completed simulations!')
        norms = self._compute_efield_magnitude(E, zero_negatives=True)
        scores = (norms * self.tw[:, None] * self.volumes[:, None]).sum(axis=0)
        return scores

    def visualize_evaluate(self,
                           x=None,
                           y=None,
                           theta=None,
                           matsimnibs=None,
                           out_sim=None,
                           out_geo=None):
        '''
        Generate a visualization of a TMS simulation on a given
        coordinate

        Arguments:
            x (float): Sampling domain x
            y (float): Sampling domain y
            theta (float): Sampling domain rotation angle
            matsimnibs (ndarray): (4,4) Matsimnibs matrix
            out_sim (str): Output path for msh file
            out_geo (str): Output path for coil position file
            fields (List[str]): Fields to include in visualization
                [Default: 'e']

        Note:
            If both :math:`(x,y,theta)` and `matsimnibs` are provided,
                the former takes precedence.

        Raises:
            ValueError if either a full :math:`(x,y,theta)` set of inputs
                or `matsimnibs` is not provided
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
                                                 fn_geo=out_geo)
        ind = _get_elmdata_index(res.elmdata, 'E')
        E = res.elmdata.pop(ind)
        magnitude = self._compute_efield_magnitude(E.value,
                                                   zero_negatives=True)

        if self.direction is None:
            label = 'normE'
        else:
            label = 'direction_normE'
        res.add_element_field(magnitude, label)

        if np.any(np.where(self.tw)[0]):
            logger.info("Appending weightfunction to output")
            wf_field = np.zeros_like(res.elmdata[0].value)
            wf_field[self.simulator.roi] = self.tw
            res.add_element_field(wf_field, 'weightfunction')
        else:
            logger.warning("Weightfunction was not set or is all"
                           " zeros! Not displaying in final result")

        mesh_io.write_msh(res, out_sim)
        logger.info(f"Wrote msh into {out_sim}")

    def get_coil2cortex_distance(self, input_coord):
        '''
        Compute distance from coil to cortical surface mesh using
        ray interception.

        This function wraps: `simnibs.msh.Msh.intercept_ray`

        Arguments:
            input_coord (tuple): (x,y,theta) tuple describing input
                coordinates on objective function domain

        Returns:
            d (float): The distance between the proposed coil location
                and the target ROI on the cortex
        '''

        # Compute the coil affine matrix
        coil_affine = self.domain.place_coil(self.model, *input_coord)
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
        """
        Arguments:
            model (fieldopt.geometry.mesh_wrapper.HeadModel): Head model
            solver (str): Solver to use
            didt (float): Stimulation dI/dt
            coil (str): Path to TMS coil definition file
            roi (ndarray): Element IDs to compute fields on
            num_workers (int): Number of workers to use for setting up
                stimulation problem
            nthreads (int): Number of threads for solver to use
        """

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
        Construct right hand side of the TMS matrix problem

        Arguments;
            mesh (simnibs.msh.mesh_io.Msh): Head model
            m (ndarray): (4,4) Matsimnibs matrix

        Returns:
            b (ndarray): (N,) RHS of simulation
            dadt (ndarray): dA/dt for a given TMS coil position
        '''

        dadt = coil_lib.set_up_tms(mesh,
                                   self.coil,
                                   m,
                                   self.didt,
                                   fn_geo=fn_geo)
        b = self.FEMSystem.assemble_tms_rhs(dadt)
        return b, dadt

    def calc_E(self, V, dAdt):
        '''
        Compute TMS electric fields from potentials

        Arguments:
            V (ndarray): TMS field potentials
            dAdt (ndarray): Magnetic vector potential time derivative
        '''
        E = np.stack([-d.dot(V) for d in self.D], axis=1) * 1e3
        E -= dAdt
        return E

    def solve(self, B):
        '''
        Solve FEM

        Arguments:
            B (ndarray): Right hand-side of TMS FEM equation

        Note:
            The solver stores the pre-computed A matrix of the left-hand
            side
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
        '''
        Run a TMS simulation on mesh with a set of coil positions
        defined by `matsimnibs`

        Arguments:
            mesh (simnibs.msh.Msh): Head model
            matsimnibs (List[ndarray]): List of (4,4) matsimnibs matrices

        Returns:
            E (ndarray): (T,3,M) Electric field matrix (T,3) for M
                TMS position problems.
        '''

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
        '''
        Compute TMS field simulation for the full head model

        Arguments:
            mesh (simnibs.msh.Msh): Head model
            matsimnibs (ndarray): (4,4) matsimnibs matrix
            fields (List[str]): List of fields to calculate
            fn_geo (str): Path to output .geo coil position file
        '''

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


def _get_elmdata_index(elmdata_arr, fieldname):
    try:
        return [
            i for i, ed in enumerate(elmdata_arr) if ed.field_name == fieldname
        ][0]
    except IndexError:
        raise
