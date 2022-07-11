'''Sampling domains for TMS field optimization
'''

import fieldopt.geometry.geometry as geometry
import fieldopt.geometry.mesh as mg
import numpy as np
import numpy.linalg as nplalg


class QuadraticDomain():
    def __init__(self, mesh, initial_p, span=35, local_span=8, distance=1):
        '''
        Build a quadratic sampling domain on a Mesh around point `initial_p`

        Arguments:
            mesh: `fieldopt.geometry.mesh_wrapper.HeadModel` object
            initial_p (ndarray): 1x3 Initial point to grow sampling region
            span (float): Radius of points to include in sampling surface
            local_span (float): Radius of points to include in construction
                of local geometry for normal and curvature estimation
            distance (float): Distance from coil to head surface
        '''

        self.span = span
        self.local_span = local_span
        self.distance = distance
        self.C, self.iR, self.bounds = self._initialize_quadratic_surface(
            mesh, initial_p)

    def _initialize_quadratic_surface(self, mesh, p):
        '''
        Construct quadratic basis and rotation at centroid point
        to use for continous sampling in head domain

        Arguments:
            mesh: `fieldopt.geometry.mesh_wrapper.HeadModel` object
                representing gmsh .msh head model file
            p (ndarray): (3,) coordinates of initial centroid point to build
                domain around

        Returns:
            C (ndarray): (6,) Coefficients of quadratic domain
            iR (ndarray): (4,4) Affine rotation matrix
            bounds (ndarray): (3,2) Bounds for (x,y,theta)
        '''

        v = mg.closest_point2surf(p, mesh.coords)
        C, R, iR = self._construct_local_quadric(mesh, v)

        # Calculate neighbours, rotate to flatten on XY plane
        neighbours_ind = np.where(
            nplalg.norm(mesh.coords - v, axis=1) < self.span)
        neighbours = mesh.coords[neighbours_ind]
        r_neighbours = geometry.affine(R, neighbours)
        minarr = np.min(r_neighbours, axis=0)
        maxarr = np.max(r_neighbours, axis=0)

        bounds = np.c_[minarr.T, maxarr.T]

        return C, iR, bounds

    def _construct_local_quadric(self, mesh, p):
        '''
        Given a single point construct a local quadric
        surface on a given mesh

        Arguments:
            mesh: `fieldopt.geometry.mesh_wrapper.HeadModel` object
                representing gmsh .msh head model file
            p (ndarray): (3,) coordinates of sampling point to build
                local domain around
        '''

        # Get local neighbourhood
        neighbours_ind = np.where(
            nplalg.norm(mesh.coords - p, axis=1) < self.local_span)

        neighbours = mesh.coords[neighbours_ind]

        # Calculate normals
        normals = mg.get_normals(mesh.nodes[neighbours_ind], mesh.nodes,
                                 mesh.coords, mesh.trigs)

        # Usage average of normals for alignment
        n = normals / nplalg.norm(normals)

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

    def _get_sample(self, mesh, x, y):
        pp = geometry.quadratic_surf_position(x, y, self.C)[np.newaxis, :]
        p = geometry.affine(self.iR, pp)
        v = mg.closest_point2surf(p, mesh.coords)
        C, _, iR = self._construct_local_quadric(mesh, v)
        _, _, n = geometry.compute_principal_dir(0, 0, C)

        # Map normal to coordinate space
        n_r = iR[:3, :3] @ n
        n_r = n_r / nplalg.norm(n_r)

        # Push sample out by set distance
        sample = v + (n_r * self.distance)

        return sample, iR, C, n

    def place_coil(self, mesh, x, y, theta):
        '''
        Place coil on mesh surface

        Arguments:
            mesh: `fieldopt.geometry.mesh_wrapper.HeadModel` object
            x (float): x coordinate of domain
            y (float): y coordinate of domain
            theta (float): Coil orientation
            flip_norm (bool): Whether the normal should be flipped
                when constructing the orientation matrix

        Returns:
            matsimnibs (ndarray): A matsimnibs orientation matrix
        '''
        sample, R, C, _ = self._get_sample(mesh, x, y)
        preaff_rot, preaff_norm = geometry.quadratic_surf_rotation(
            0, 0, theta, C)
        rot = R[:3, :3] @ preaff_rot
        n = R[:3, :3] @ preaff_norm

        return geometry.define_coil_orientation(sample, rot, -1 * n)
