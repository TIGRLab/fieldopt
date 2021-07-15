#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
from numpy.linalg import norm as vecnorm

from simnibs import cond
from simnibs.msh import mesh_io
from simnibs.simulation.fem import (FEMSystem, _set_up_global_solver,
                                    calc_fields)
import simnibs.simulation.coil_numpy as coil_lib

from fieldopt import geometry

from multiprocessing import Pool

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
    FIELDS = ['E', 'e', 'J', 'j']

    def __init__(self,
                 mesh_file,
                 initial_centroid,
                 tet_weights,
                 coil,
                 span=35,
                 local_span=8,
                 distance=1,
                 didt=1e6,
                 cpus=1,
                 solver_options=None):
        '''
        Standard constructor
        Arguments:
            mesh_file                   Path to FEM model
            initial_centroid            Initial point to grow sampling region
            tet_weights                 Weighting scores for each tetrahedron
                                        (1D array ordered by node ID)
            coil                        TMS coil file (either dA/dt volume or
                                        coil geometry)
            span                        Radius of points to include in
                                        sampling surface
            local_span                  Radius of points to include in
                                        construction of local geometry
                                        for normal and curvature estimation
            distance                    Distance from coil to head surface
            didt                        Intensity of stimulation
            cpus                        Number of cpus to use for simulation
            solver_options              Options to pass into SimNIBS to
                                        configure solver
        '''

        self.mesh = mesh_file
        self.tw = tet_weights
        self.coil = coil
        self.didt = didt
        logger.info(f"Configured to use {cpus} cpus...")
        self.cpus = cpus
        self.geo_radius = local_span
        self.distance = distance
        self.solver_opt = solver_options
        self.centroid = initial_centroid
        self.span = span

        # Control for coil file-type and SimNIBS changing convention
        if self.coil.endswith('.ccd'):
            self.normflip = 1
        else:
            self.normflip = -1

        self.FEMSystem = None
        self.initialize()

    def initialize(self):
        '''
        Initialize objective function to allow for objective
        function evaluations.

        Caches:
            - Mesh
            - Nodes, Coordinates
            - Sampling surface at given centroid
            - Conductivity list
            - FEMSystem

        Opens multiprocessing pool to have jobs submitted
        '''

        logger.info('Loading in coordinate data from mesh file...')
        self.nodes, self.coords, _ = geometry.load_gmsh_nodes(
            self.mesh, (2, 5))
        _, _, trigs = geometry.load_gmsh_elems(self.mesh, (2, 5))
        self.trigs = np.array(trigs).reshape(-1, 3)
        logger.info('Successfully pulled in node and element data!')

        # Construct basis of sampling space using centroid
        logger.info('Constructing initial sampling surface...')
        C, iR, bounds = self._initialize_quadratic_surface(
            self.centroid, self.span)
        self.C = C
        self.iR = iR
        self.bounds = bounds
        logger.info('Successfully constructed initial sampling surface')

        logger.info('Caching mesh file...')
        self.cached_mesh = mesh_io.read_msh(self.mesh)
        self.cached_mesh.fix_surface_labels()
        logger.info('Successfully cached mesh file')

        logger.info('Storing standard conductivity values...')
        condlist = [c.value for c in cond.standard_cond()]
        self.cond = cond.cond2elmdata(self.cached_mesh, condlist)
        logger.info('Successfully stored conductivity values...')

        logger.info("Initializing FEM problem")
        self.FEMSystem = FEMSystem.tms(self.cached_mesh,
                                       self.cond,
                                       solver_options=self.solver_opt)
        self._solver = self.FEMSystem._solver
        logger.info("Completed FEM initialization")
        return

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.mesh)
        print('Coil:', self.coil)
        return ''

    def get_bounds(self):
        return self.bounds

    def _initialize_quadratic_surface(self, centroid, span):
        '''
        Construct quadratic basis and rotation at centroid point
        to use for sampling
        '''

        v = geometry.closest_point2surf(centroid, self.coords)
        C, R, iR = self._construct_local_quadric(v, 0.75 * span)

        # Calculate neighbours, rotate to flatten on XY plane
        neighbours_ind = np.where(vecnorm(self.coords - v, axis=1) < span)
        neighbours = self.coords[neighbours_ind]
        r_neighbours = geometry.affine(R, neighbours)
        minarr = np.min(r_neighbours, axis=0)
        maxarr = np.max(r_neighbours, axis=0)

        bounds = np.c_[minarr.T, maxarr.T]

        return C, iR, bounds

    def _construct_local_quadric(self, p, tol=1e-3):
        '''
        Given a single point construct a local quadric
        surface on a given mesh
        '''

        # Get local neighbourhood
        neighbours_ind = np.where(vecnorm(self.coords - p, axis=1) < tol)

        neighbours = self.coords[neighbours_ind]

        # Calculate normals
        normals = geometry.get_normals(self.nodes[neighbours_ind], self.nodes,
                                       self.coords, self.trigs)

        # Usage average of normals for alignment
        n = normals / vecnorm(normals)

        # Make transformation matrix
        z = np.array([0, 0, 1])
        R = np.eye(4)
        R[:3, :3] = geometry.rotate_vec2vec(n, z)
        T = np.eye(4)
        T[:3, 3] = -p
        affine = R @ T

        # Create inverse rotation
        iR = R
        iR[:3, :3] = iR[:3, :3].T

        # Create inverse translation
        iT = T
        iT[:3, 3] = -T[:3, 3]
        i_affine = iT @ iR

        # Perform quadratic fitting
        r_neighbours = geometry.affine(affine, neighbours)
        C = geometry.quad_fit(r_neighbours[:, :2], r_neighbours[:, 2])

        return C, affine, i_affine

    def _construct_sample(self, x, y):
        '''
        Given a sampling point, estimate local geometry to
        get accurate normals/curvatures
        '''

        pp = geometry.map_param_2_surf(x, y, self.C)[np.newaxis, :]
        p = geometry.affine(self.iR, pp)
        v = geometry.closest_point2surf(p, self.coords)
        C, _, iR = self._construct_local_quadric(v, self.geo_radius)
        _, _, n = geometry.compute_principal_dir(0, 0, C)

        # Map normal to coordinate space
        n_r = iR[:3, :3] @ n
        n_r = n_r / vecnorm(n_r)

        # Push sample out by set distance
        sample = v + (n_r * self.distance)

        return sample, iR, C, n

    def _transform_input(self, x, y, theta):
        '''
        Generates a coil orientation matrix given inputs from a
        quadratic surface sampling domain
        '''

        sample, R, C, _ = self._construct_sample(x, y)
        preaff_rot, preaff_norm = geometry.map_rot_2_surf(0, 0, theta, C)
        rot = R[:3, :3] @ preaff_rot
        n = R[:3, :3] @ preaff_norm

        o_matrix = geometry.define_coil_orientation(sample, rot,
                                                    self.normflip * n)
        return o_matrix

    def _simulate(self, matsimnib, didt, out_geo=None, out_sim=None):
        '''
        Re-implementation of SimNIBS _run_tms function to return
        mesh object rather than to write to a file
        '''

        # SimNIBS core routine
        dAdt = coil_lib.set_up_tms(self.cached_mesh,
                                   self.coil,
                                   matsimnib,
                                   didt,
                                   fn_geo=out_geo)
        b = self.FEMSystem.assemble_tms_rhs(dAdt)
        v = self.FEMSystem.solve(b)
        v = mesh_io.NodeData(v, name='v', mesh=self.cached_mesh)
        v.mesh = self.cached_mesh
        res = calc_fields(v, self.FIELDS, cond=self.cond, dadt=dAdt)

        if out_sim:
            mesh_io.write_msh(res, out_sim)

        # Extract scores
        return self._calculate_score(res)

    def _prepare_tms_matrix(self, m):
        '''
        Construct right hand side of the TMS linear
        problem
        '''

        # Compute dA/dt for coil posiion on mesh
        dadt = coil_lib.set_up_tms(self.cached_mesh, self.coil,
                m, self.didt)
        dof_map = copy.deepcopy(self.FEMSystem.dof_map)

        # Assemble b
        b = self.FEMSystem.assemble_tms_rhs(dadt)
        return self.FEMSystem.dirchlet.apply_to_rhs(
                self.FEMSystem.A,
                b,
                dof_map
        )

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
        logger.info('Constructing right-hand side of FEM...')
        start = time.time()
        with Pool(processes=self.cpus) as pool:
            B = pool.apply(self._prepare_tms_matrix, matsimnibs)
        end = time.time()
        logger.info(f"Took {end-start:.2f}")

        # Each column vector is a TMS coil position
        V = self._solver.solve(B)
        
        # Need a more efficient routine to compute scores


        logger.info('Successfully completed simulations!')
        return res

    def _get_tet_ids(self, entity):
        '''
        Pull list of element IDs for a given gmsh entity
        '''

        tet_ids = np.where(self.cached_mesh.elm.tag1 == entity[1])
        return tet_ids

    def _calculate_score(self, sim_msh):
        '''
        Given a simulation output file, compute the score

        Volumes are integrated into the score function to deal with
        score inflation due to a larger number of elements near
        more complex geometry.
        '''

        tet_ids = self._get_tet_ids(self.FIELD_ENTITY)
        normE = sim_msh.elmdata[1].value[tet_ids]

        neg_ind = np.where(normE < 0)
        normE[neg_ind] = 0

        vols = self.cached_mesh.elements_volumes_and_areas()[tet_ids]

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
        matsimnibs = [self._transform_input(x, y, t) for x, y, t in input_list]

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
        coil_affine = self._transform_input(*input_coord)
        n = coil_affine[:3, 2]
        p0 = coil_affine[:3, 3]
        p1 = p0 + (n * 200)

        # Calculate interception with pial surface
        pial_mesh = self.cached_mesh.crop_msh(self.PIAL_ENTITY[1],
                                              self.PIAL_ENTITY[0])
        _, pos = pial_mesh.intercept_ray(p0, p1)

        # Compute distance
        return np.linalg.norm(p0 - pos)
