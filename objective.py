#!/usr/bin/env python
# coding: utf-8

import os
import tempfile
import numpy as np
from numpy.linalg import norm as vecnorm
from simnibs import cond
from simnibs.msh import mesh_io
from simnibs.simulation.fem import tms_coil
from fieldopt import geolib
import logging
from shutil import copytree, move

logging.basicConfig(
    format='[%(levelname)s - %(name)s.%(funcName)5s() ] %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldFunc():

    '''
    This class provides an interface in which the details related
    to simulation and score extraction is abstracted away for
    any general Bayesian Optimization package

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
                 field_dir,
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
            field_dir                   Directory to perform simulation
                                        experiments in
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
        '''

        self.mesh = mesh_file
        self.tw = tet_weights
        self.field_dir = field_dir
        self.coil = coil
        self.didt = didt

        logger.info(f"Configured to use {cpus} cpus...")
        self.cpus = cpus
        self.geo_radius = local_span
        self.distance = distance
        self.solver_opt = solver_options

        logger.info('Loading in coordinate data from mesh file...')
        self.nodes, self.coords, _ = geolib.load_gmsh_nodes(self.mesh, (2, 5))
        _, _, trigs = geolib.load_gmsh_elems(self.mesh, (2, 5))
        self.trigs = np.array(trigs[0]).reshape(-1, 3)
        logger.info('Successfully pulled in node and element data!')

        # Construct basis of sampling space using centroid
        logger.info('Constructing initial sampling surface...')
        C, iR, bounds = self._initialize(initial_centroid, span)
        self.C = C
        self.iR = iR
        self.bounds = bounds
        logger.info('Successfully constructed initial sampling surface')

        # Store single read in memory, this will prevent GC issues
        logger.info('Caching mesh file on instance construction...')
        self.cached_mesh = mesh_io.read_msh(mesh_file)
        self.cached_mesh.fix_surface_labels()
        logger.info('Successfully cached mesh file')

        logger.info('Storing standard conductivity values...')
        condlist = [c.value for c in cond.standard_cond()]
        self.cond = cond.cond2elmdata(self.cached_mesh, condlist)
        logger.info('Successfully stored conductivity values...')

        # Control for coil file-type and SimNIBS changing convention
        if self.coil.endswith('.ccd'):
            self.normflip = 1
        else:
            self.normflip = -1

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.mesh)
        print('Coil:', self.coil)
        print('Field Directory:', self.field_dir)
        return ''

    def get_bounds(self):
        return self.bounds

    def _initialize(self, centroid, span):
        '''
        Construct quadratic basis and rotation at centroid point
        to use for sampling
        '''

        v = geolib.closest_point2surf(centroid, self.coords)
        C, R, iR = self._construct_local_quadric(v, 0.75 * span)

        # Calculate neighbours, rotate to flatten on XY plane
        neighbours_ind = np.where(vecnorm(self.coords - v, axis=1) < span)
        neighbours = self.coords[neighbours_ind]
        r_neighbours = geolib.affine(R, neighbours)
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
        normals = geolib.get_normals(self.nodes[neighbours_ind], self.nodes,
                                     self.coords, self.trigs)

        # Usage average of normals for alignment
        n = np.mean(normals, axis=0)
        n = n / vecnorm(n)

        # Make transformation matrix
        z = np.array([0, 0, 1])
        R = np.eye(4)
        R[:3, :3] = geolib.rotate_vec2vec(n, z)
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
        r_neighbours = geolib.affine(affine, neighbours)
        C = geolib.quad_fit(r_neighbours[:, :2], r_neighbours[:, 2])

        return C, affine, i_affine

    def _construct_sample(self, x, y):
        '''
        Given a sampling point, estimate local geometry to
        get accurate normals/curvatures
        '''

        pp = geolib.map_param_2_surf(x, y, self.C)[np.newaxis, :]
        p = geolib.affine(self.iR, pp)
        v = geolib.closest_point2surf(p, self.coords)
        C, _, iR = self._construct_local_quadric(v, self.geo_radius)
        _, _, n = geolib.compute_principal_dir(0, 0, C)

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
        preaff_rot, preaff_norm = geolib.map_rot_2_surf(0, 0, theta, C)
        rot = R[:3, :3] @ preaff_rot
        n = R[:3, :3] @ preaff_norm

        o_matrix = geolib.define_coil_orientation(sample, rot, self.normflip*n)
        return o_matrix

    def _get_simulation_outnames(self, num_sims, sim_dir):

        simu_name = os.path.join(sim_dir, 'TMS_{}'.format(1))
        coil_name = os.path.splitext(os.path.basename(self.coil))[0]

        fn_simu = [
            "{0}-{1:0=4d}_{2}_".format(simu_name, i + 1, coil_name)
            for i in range(num_sims)
        ]
        output_names = [f + 'scalar.msh' for f in fn_simu]
        geo_names = [f + 'coil_pos.geo' for f in fn_simu]

        return output_names, geo_names

    def _run_simulation(self, matsimnibs, sim_dir):

        if not isinstance(matsimnibs, list):
            matsimnibs = [matsimnibs]

        # Construct standard inputs
        logger.info('Constructing inputs for simulation...')
        logger.info(f'Using didt={self.didt}')
        didt_list = [self.didt] * len(matsimnibs)

        output_names, geo_names = self._get_simulation_outnames(
                len(matsimnibs), sim_dir)

        logger.info('Starting SimNIBS simulations...')
        tms_coil(self.cached_mesh,
                 self.cond,
                 self.coil,
                 self.FIELDS,
                 matsimnibs,
                 didt_list,
                 output_names,
                 geo_names,
                 n_workers=self.cpus,
                 solver_options=self.solver_opt)
        logger.info('Successfully completed simulations!')

        return sorted(output_names)

    def run_simulation(self, coord, out_sim, out_geo):
        '''
        Given a quadratic surface input (x,y) and a rotational
        interpolation angle (theta) run a simulation and
        save the resulting output into <out_dir>

        Arguments:
            [(x,y,theta),...]           A iterable of iterable (x,y,theta)

        Returns:
            sim_file                    Path to simulation file
                                        in output directory
            score                       Score
        '''

        logger.info('Transforming inputs...')
        matsimnibs = self._transform_input(*coord)

        # Run simulation in a temporary directory
        with tempfile.TemporaryDirectory(dir=self.field_dir) as sim_dir:
            logger.info('Running simulation')
            sim_file = self._run_simulation(matsimnibs, sim_dir)[0]

            logger.info('Calculating Score...')
            scores = self._calculate_score(sim_file)
            logger.info('Successfully pulled scores!')

            sim_names, geo_names = self._get_simulation_outnames(
                    1, sim_dir)

            # Transfer files to destination
            logger.info('Transferring files to destination')
            move(geo_names[0], out_geo)
            move(sim_names[0], out_sim)
            logger.info('Succesfully transferred files')

        return scores, matsimnibs

    def _calculate_score(self, sim_file):
        '''
        Given a simulation output file, compute the score
        '''

        logger.info('Loading gmsh elements from {}...'.format(sim_file))
        _, elem_ids, _ = geolib.load_gmsh_elems(sim_file, self.FIELD_ENTITY)

        logger.info('Pulling field values from {}...'.format(sim_file))
        normE = geolib.get_field_subset(sim_file, elem_ids)
        logger.info('Successfully pulled field values!')

        neg_ind = np.where(normE < 0)
        normE[neg_ind] = 0

        return np.dot(self.tw, normE)

    def evaluate(self, input_list, out_dir=None):
        '''
        Given a quadratic surface input (x,y) and rotational
        interpolation angle (theta) compute the resulting field score
        over a region of interest
        Arguments:
            [(x,y,theta),...]           A iterable of iterable (x,y,theta)

        Returns:
            scores                      An array of scores in order of inputs
        '''

        with tempfile.TemporaryDirectory(dir=self.field_dir) as sim_dir:

            logger.info('Transforming inputs...')
            matsimnibs = [
                self._transform_input(x, y, t) for x, y, t in input_list
            ]

            logger.info('Running simulations...')
            sim_files = self._run_simulation(matsimnibs, sim_dir)

            logger.info('Calculating Scores...')
            scores = np.array([self._calculate_score(s) for s in sim_files])

            logger.info('Successfully pulled scores!')

            if out_dir is not None:
                logger.info('Storing outputs!')
                copytree(sim_dir, out_dir, dirs_exist_ok=True)
                logger.info(f'Successfully stored outputs in {out_dir}')

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
