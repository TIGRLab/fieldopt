#!/usr/bin/env python
# coding: utf-8

import os
import tempfile
import numpy as np
from simnibs import cond
from simnibs.msh import mesh_io
from simnibs.simulation.fem import tms_coil
from fieldopt import geolib


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
                 quad_surf_consts,
                 surf_to_mesh_matrix,
                 tet_weights,
                 field_dir,
                 coil,
                 didt=1e6,
                 cpus=1):
        '''
        Standard constructor
        Arguments:
            mesh_file                   Path to FEM model
            quad_surf_consts            Quadratic surface constants
            surf_to_mesh_matrix         (3,3) affine transformation matrix
            tet_weights                 Weighting scores for each tetrahedron
                                        (1D array ordered by node ID)
            field_dir                   Directory to perform simulation
                                        experiments in
            coil                        TMS coil file (either dA/dt volume or
                                        coil geometry)
        '''

        self.mesh = mesh_file
        self.C = quad_surf_consts
        self.iR = surf_to_mesh_matrix
        self.tw = tet_weights
        self.field_dir = field_dir
        self.coil = coil
        self.didt = didt
        self.cpus = cpus

        # Store single read in memory, this will prevent GC issues
        # and will force only a single slow read of the file
        self.cached_mesh = mesh_io.read_msh(mesh_file)
        self.cached_mesh.fix_surface_labels()
        condlist = [c.value for c in cond.standard_cond()]
        self.cond = cond.cond2elmdata(self.cached_mesh, condlist)

    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.mesh)
        print('Coil:', self.coil)
        print('Field Directory:', self.field_dir)
        return ''

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

        # Construct standard inputs
        didt_list = [self.didt] * len(matsimnibs)
        simu_name = os.path.join(sim_dir, 'TMS_{}'.format(1))
        coil_name = os.path.splitext(os.path.basename(self.coil))[0]

        fn_simu = [
            "{0}-{1:0=4d}_{2}_".format(simu_name, i + 1, coil_name)
            for i in range(len(matsimnibs))
        ]
        output_names = [f + 'scalar.msh' for f in fn_simu]
        geo_names = [f + 'coil_pos.geo' for f in fn_simu]

        tms_coil(self.cached_mesh,
                 self.cond,
                 self.coil,
                 self.FIELDS,
                 matsimnibs,
                 didt_list,
                 output_names,
                 geo_names,
                 n_workers=self.cpus)

        return sorted(output_names)

    def _calculate_score(self, sim_file):
        '''
        Given a simulation output file, compute the score
        '''

        _, elem_ids, _ = geolib.load_gmsh_elems(sim_file, self.FIELD_ENTITY)
        normE = geolib.get_field_subset(sim_file, elem_ids)
        return np.dot(self.tw, normE)

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

            matsimnibs = [
                self._transform_input(x, y, t) for x, y, t in input_list
            ]
            sim_files = self._run_simulation(matsimnibs, sim_dir)
            scores = np.array([self._calculate_score(s) for s in sim_files])

        return scores
