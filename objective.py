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
            distance                    Distance from coil to head surface
        '''

        self.mesh = mesh_file
        self.centroid = initial_centroid

        logger.info('Constructing initial sampling surface')
        self.C, self.iR = self._initialize(span)
        logger.info('Successfully constructed initial sampling surface')

        self.tw = tet_weights
        self.field_dir = field_dir
        self.coil = coil
        self.didt = didt
        self.cpus = cpus

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

    def _initialize(self, span):
        '''
        Construct quadratic basis and rotation at centroid point
        to use for sampling
        '''

        # Step 1: Read nodes and elements from mesh
        n_tag, n_coord, _ = geolib.load_gmsh_nodes(self.mesh, (2, 5))
        _, _, tri = geolib.load_gmsh_elems(self.mesh, (2, 5))

        # Step 2: Get closest point to surface
        v_centre = geolib.closest_point2surf(self.centroid, n_coord)

        # Step 3: Span nearest points
        neighbours_ind = np.where(vecnorm(n_coord - v_centre, axis=1) < span)
        neighbours = n_coord[neighbours_ind]

        # Step 4: Calculate normals of neigbouring values
        normals = geolib.get_normals(neighbours_ind, n_tag, n_coord, tri)

        # Step 5: Use average normal
        n = np.mean(normals, axis=0)
        n = n / vecnorm(n)

        # Step 6: Perform quadratic fitting procedure
        z = np.array([0, 0, 1])
        R = geolib.rotate_vec2vec(n, z)
        r_neighbours = (R @ neighbours.T).T
        C = geolib.quad_fit(r_neighbours[:, :2], r_neighbours[:, 2])

        return R.T, C


    def _transform_input(self, x, y, theta):
        '''
        Generates a coil orientation matrix given inputs from a
        quadratic surface sampling domain
        '''

        preaff_loc = geolib.map_param_2_surf(x, y, self.C)
        preaff_rot, preaff_norm = geolib.map_rot_2_surf(x, y, theta, self.C)

        loc = np.matmul(self.iR, preaff_loc)
        rot = np.matmul(self.iR, preaff_rot)
        n = np.matmul(self.iR, preaff_norm)

        o_matrix = geolib.define_coil_orientation(loc, rot, n)
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
