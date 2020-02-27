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

logging.basicConfig(
    format='[%(levelname)s - %(name)s.%(funcName)5s() ] %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldFunc():

    FIELD_ENTITY = (3, 2)
    FIELDS = ['E', 'e', 'J', 'j']
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
    def __init__(self,
                 mesh_file,
                 initial_centroid,
                 span,
                 tet_weights,
                 field_dir,
                 coil,
                 span=35,
                 local_span=3,
                 distance=1,
                 didt=1e6,
                 cpus=1):
        '''
        Standard constructor
        Arguments:
            mesh_file                   Path to FEM model
            initial_centroid            Initial point to grow sampling region
            quad_surf_consts            Quadratic surface constants
            surf_to_mesh_matrix         (3,3) affine transformation matrix
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
        '''

        self.mesh = mesh_file
        self.tw = tet_weights
        self.field_dir = field_dir
        self.coil = coil
        self.didt = didt
        self.cpus = cpus
        self.geo_radius = local_span

        logger.info('Loading in coordinate data from mesh file...')
        self.nodes, self.coords = geolib.load_gmsh_nodes(self.mesh, (2, 5))
        _, _, self.trigs = geolib.load_gmsh_elems(self.mesh, (2, 5))
        logger.info('Successfully pulled in node and element data!')

        # Construct basis of sampling space using centroid
        logger.info('Constructing initial sampling surface...')
        self.C, self.iR, self.bounds = self._initialize(initial_centroid, span)
        logger.info('Successfully constructed initial sampling surface')

        # Store single read in memory, this will prevent GC issues
        # and will force only a single slow read of the file
        logger.info('Caching mesh file on instance construction...')
        self.cached_mesh = mesh_io.read_msh(mesh_file)
        self.cached_mesh.fix_surface_labels()
        logger.info('Successfully cached mesh file')

        logger.info('Storing standard conductivity values...')
        condlist = [c.value for c in cond.standard_cond()]
        self.cond = cond.cond2elmdata(self.cached_mesh, condlist)
        logger.info('Successfully stored conductivity values...')

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.mesh)
        print('Coil:', self.coil)
        print('Field Directory:', self.field_dir)
        return ''

    @property
    def bounds(self):
        return self.bounds

    def _initialize(self, centroid, span):
        '''
        Construct quadratic basis and rotation at centroid point
        to use for sampling
        '''

        v = geolib.closest_point2surf(centroid, self.coords)
        C, R, _ = self._construct_local_quadric(v)

        # Calculate neighbours, rotate to flatten on XY plane
        neighbours_ind = np.where(vecnorm(self.coords - v, axis=1) < span)
        neighbours = self.coords[neighbours_ind]
        r_neighbours = (R @ neighbours.T).T
        minarr = np.min(r_neighbours, axis=0)
        maxarr = np.max(r_neighbours, axis=0)

        bounds = np.c_[minarr.T, maxarr.T]

        return C, R.T, bounds

    def _construct_local_quadric(self, p):
        '''
        Given a single point construct a local quadric
        surface on a given mesh
        '''

        # Get local neighbourhood
        neighbours_ind = np.where(
            vecnorm(self.coords - p, axis=1) < self.geo_radius)
        neighbours = self.coord[neighbours_ind]

        # Calculate normals
        normals = geolib.get_normals(neighbours, self.nodes, self.coord,
                                     self.trigs)

        # Usage average of normals
        n = np.mean(normals, axis=0)
        n = n / vecnorm(n)

        # Perform quadratic fitting
        z = np.array([0, 0, 1])
        R = geolib.rotate_vec2vec(n, z)
        r_neighbours = (R @ neighbours.T).T
        C = geolib.quad_fit(r_neighbours[:, :2], r_neighbours[:, 2])

        return C, R, n

    def _construct_sample(self, x, y):
        '''
        Given a sampling point, estimate local geometry to
        get accurate normals/curvatures
        '''

        p = self.iR @ geolib.map_param_2_surf(x, y, self.C)
        v = geolib.closest_point2surf(p, self.coords)
        C, R, n = self._construct_local_quadric(v)

        # Need to invert transformation
        R = R.T

        # Push closest point by normal
        n_r = R @ n
        n_r = n_r / vecnorm(n_r)

        # Push sample out by set distance
        sample = p + (n_r * self.distance)

        return sample, R, C

    def _transform_input(self, x, y, theta):
        '''
        Generates a coil orientation matrix given inputs from a
        quadratic surface sampling domain
        '''

        sample, R, C = self._construct_sample(x, y)
        preaff_rot, preaff_norm = geolib.map_rot_2_surf(0, 0, theta, C)
        rot = R @ preaff_rot
        n = R @ preaff_norm

        o_matrix = geolib.define_coil_orientation(sample, rot, n)
        return o_matrix

    def _run_simulation(self, matsimnibs, sim_dir):

        if not isinstance(matsimnibs, list):
            matsimnibs = [matsimnibs]

        # Construct standard inputs
        logger.info('Constructing inputs for simulation...')
        logger.info(f'Using didt={self.didt}')
        didt_list = [self.didt] * len(matsimnibs)
        simu_name = os.path.join(sim_dir, 'TMS_{}'.format(1))
        coil_name = os.path.splitext(os.path.basename(self.coil))[0]

        fn_simu = [
            "{0}-{1:0=4d}_{2}_".format(simu_name, i + 1, coil_name)
            for i in range(len(matsimnibs))
        ]
        output_names = [f + 'scalar.msh' for f in fn_simu]
        geo_names = [f + 'coil_pos.geo' for f in fn_simu]

        logger.info('Starting SimNIBS simulations...')
        tms_coil(self.cached_mesh,
                 self.cond,
                 self.coil,
                 self.FIELDS,
                 matsimnibs,
                 didt_list,
                 output_names,
                 geo_names,
                 n_workers=self.cpus)
        logger.info('Successfully completed simulations!')

        return sorted(output_names)

    def _calculate_score(self, sim_file):
        '''
        Given a simulation output file, compute the score
        '''

        logger.info('Loading gmsh elements from {}...'.format(sim_file))
        _, elem_ids, _ = geolib.load_gmsh_elems(sim_file, self.FIELD_ENTITY)

        logger.info('Pulling field values from {}...'.format(sim_file))
        normE = geolib.get_field_subset(sim_file, elem_ids)
        logger.info('Successfully pulled field values!')

        return np.dot(self.tw, normE)

    def run_simulation(self, coord, out_dir):
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
        matsimnibs = self._transform_input(coord[0], coord[1], coord[2])

        logger.info('Running simulation')
        sim_file = self._run_simulation(matsimnibs, out_dir)[0]

        logger.info('Calculating Score...')
        scores = self._calculate_score(sim_file)
        logger.info('Successfully pulled scores!')

        return sim_file, scores

    def evaluate(self, input_list):
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

        return scores
