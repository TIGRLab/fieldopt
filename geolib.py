#!/usr/bin/env python
# coding: utf-8

# Library of useful geometry functions

import numpy as np
from scipy.linalg import lstsq
from numpy.linalg import norm as vecnorm
import gmsh
import numba


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product
    matrix for vector. The skew symmetric cross product matrix is defined
    such that:
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross
        product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    Source:
    https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array
    """
    if isinstance(vector, np.ndarray):
        return np.array([[0, -vector.item(2),
                          vector.item(1)],
                         [vector.item(2), 0, -vector.item(0)],
                         [-vector.item(1), vector.item(0), 0]])

    return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def affine(A, x):
    '''
    Apply affine transformation A to vector set x
    '''

    h = np.c_[x, np.ones((x.shape[0], 1))]
    Ah = A @ h.T
    return Ah.T[:, :3]


def rotate_vec2vec(v1, v2):
    '''
    Rotate vector v1 --> v2 and return the transformation matrix R that
    achieves this
    '''

    n = np.cross(v1, v2)
    sinv = vecnorm(n)
    cosv = np.dot(v1, v2)
    R = np.eye(3) + skew(n) + np.matmul(skew(n), skew(n)) * (1 - cosv) / (sinv
                                                                          **2)
    return R


def closest_point2surf(p, coords):

    ind = np.argmin(vecnorm(coords - p, axis=1))
    return coords[ind, :]


def quad_fit(X, b):
    '''
    Perform quadratic surface fitting of form:
    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
    By finding the least squares solution of Ax = b
    '''

    # Formulate the linear problem
    A = np.c_[np.ones((X.shape[0], 1)), X[:, :2],
              np.prod(X[:, :2], axis=1), X[:, :2]**2]

    C, _, _, _ = lstsq(A, b)
    return C


def compute_principal_dir(x, y, C):
    '''
    Compute the principal direction of a quadratic surface of form:
    S: f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
    Using the second fundamental form basis matrix eigendecomposition method

    x -- scalar x input
    y -- scalar y input
    C -- scalar quadratic vector (a,b,c,d,e,f)

    V[:,0] -- major principal direction
    V[:,1] -- minor principal direction
    n      -- normal to surface S at point (x,y)
    '''

    # Compute partial first and second derivatives
    r_x = np.array([1, 0, 2 * C[4] * x + C[1] + C[3] * y])
    r_y = np.array([0, 1, 2 * C[5] * y + C[2] + C[3] * x])
    r_xx = np.array([0, 0, 2 * C[4]])
    r_yy = np.array([0, 0, 2 * C[5]])
    r_xy = np.array([0, 0, C[3]])

    # Compute surface point normal
    r_x_cross_y = np.cross(r_x, r_y)
    n = r_x_cross_y / vecnorm(r_x_cross_y)

    # Compute second fundamental form constants
    L = np.dot(r_xx, n)
    M = np.dot(r_xy, n)
    N = np.dot(r_yy, n)

    # Form basis matrix
    P = np.array([[L, M], [M, N]])

    # Eigendecomposition, then convert into 3D vector
    _, V = np.linalg.eig(P)
    V = np.concatenate((V, np.zeros((1, 2))), axis=0)

    return V[:, 0], V[:, 1], n


def interpolate_angle(u, v, t, l=90.0):
    '''
    Perform cosine angle interpolation between two orthogonal vectors u and v

    u -- 3D vector
    v -- 3D vector orthogonal to u
    t -- interpolation value
    l -- period of rotation
    '''

    theta = (t / l) * (np.pi / 2)
    p = np.r_[u * np.cos(theta) + v * np.sin(theta), 0]
    return p


def quadratic_surf(x, y, C):
    '''
    Compute a quadratic function of form:
                    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2

    x -- scalar x input
    y -- scalar y input
    C -- quadratic constants vector (a,b,c,d,e,f)
    '''

    return C[
        0] + C[1] * x + C[2] * y + C[3] * x * y + C[4] * x * x + C[5] * y * y


def map_param_2_surf(x, y, C):
    '''
    For some mesh-based surface S, define a parameterization using a quadratic
    fit:
                    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2

    Compute the mapping from (x,y) --> (x,y,f(x,y))

    x -- scalar x input
    y -- scalar y input
    C -- scalar quadratic constants vector (a,b,c,d,e,f)
    '''

    # Compute approximate surface at (x,y)
    z = quadratic_surf(x, y, C)

    # Form input vector
    v = np.array([x, y, z], dtype=np.float64)

    return v


def map_rot_2_surf(x, y, t, C):
    '''
    For some mesh-based surface S, define a least squares quadratic
    surface parameterization S':
                        f(x,y) = a + bx + cy + dxy + ex^2 + fy^2

    Compute a mapping from (x,y,z,t) --> p which is an interpolated direction
    vector between the two principal directions on S' using t:[0,180].
    Both principal directions are defined on the (x,y) plane so we need to
    align the orientation vector p to the surface normal. This is done
    using the following steps:
    1. Find rotation R that maps the standard basis z axis
        to the normal of surface S' at point (x,y)

    2. Apply the rotation to vector p to align it to the
        new basis defined by the normal yielding p'

    x -- scalar x input
    y -- scalar y input
    t -- interpolation angle [0,180] between 2 principal directions
    C -- scalar quadratic constants vector (a,b,c,d,e,f)

    '''

    v1, v2, n = compute_principal_dir(x, y, C)
    p = interpolate_angle(v1[:2], v2[:2], t)

    z = np.array([0, 0, 1], dtype=np.float64)
    R = rotate_vec2vec(z, n)
    pp = np.matmul(R, p)

    return pp, n


def load_gmsh_nodes(gmshpath, entity):
    '''
    Given a fullpath to some .msh file
    load in the mesh nodes IDs, triangles and coordinates.

    gmshpath -- path to gmsh file
    dimtag   -- tuple specifying the (dimensionality,tagID) being loaded


    If entity=(dim,tag) not provided then pull the first entity and return
    '''

    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, coords, params = gmsh.model.mesh.getNodes(entity[0], entity[1])
    coords = np.array(coords).reshape((len(coords) // 3, 3))
    gmsh.clear()

    return nodes - 1, coords, params


def load_gmsh_elems(gmshpath, entity):
    '''
    Wrapper function for loading gmsh elements
    '''

    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, elem_ids, node_maps = gmsh.model.mesh.getElements(
        entity[0], entity[1])
    gmsh.clear()

    return nodes, elem_ids[0] - 1, node_maps[0] - 1


def define_coil_orientation(loc, rot, n):
    '''
    Construct the coil orientation matrix to be used by simnibs
    loc -- center of coil
    rot -- vector pointing in handle direction (tangent to surface)
    n   -- normal vector
    '''

    y = rot / vecnorm(rot)
    z = n / vecnorm(n)
    x = np.cross(y, z)
    c = loc

    matsimnibs = np.zeros((4, 4), dtype=np.float64)
    matsimnibs[:3, 0] = x
    matsimnibs[:3, 1] = y
    matsimnibs[:3, 2] = z
    matsimnibs[:3, 3] = c
    matsimnibs[3, 3] = 1

    return matsimnibs


def compute_field_score(normE, proj_map, parcel):
    '''
    From a list of field magnitudes in <normE> compute the
    weighted sum determined by the partial parcel weightings in proj_map

    Arguments:
    normE      --  1D array of relevant field magnitudes in the order
                   of proj_map
    proj_map   --  Array of size (len(<normE>), max(parcel ID)) where each
                   row corresponds to an
                   element (pairing with <normE>) and each column corresponds
                   to a parcel ID.
    parcel     --  The parcel to compute a score over.
                   Range = [0,max(parcel_ID)]

    Output:
    score      --  A single scalar value representing the total stimulation
    '''

    parcel_map = proj_map[:, parcel]
    return np.dot(parcel_map, normE)


@numba.njit(parallel=True)
def get_relevant_triangles(verts, triangles):
    '''
    From an array of vertices and triangles,
    get triangles that contain at least 1 vertex
    Arguments:
        verts                               1-D array of vertex IDs
        triangles                           (NX3) array of triangles, where
                                            each column is a vertex
    Output:
        t_arr                               True of triangle contains at least
                                            one vertex from list
    '''

    t_arr = np.zeros((triangles.shape[0]), dtype=np.int64)

    for t in numba.prange(0, triangles.shape[0]):
        for c in np.arange(0, 3):
            for v in verts:

                if triangles[t][c] == v:
                    t_arr[t] = 1
                    break

            if t_arr[t] == 1:
                break

    return t_arr


@numba.njit
def unitize_arr(arr):
    '''
    Normalize array row-wise
    '''

    narr = np.zeros((arr.shape[0], 3), dtype=np.float64)

    for i in np.arange(0, arr.shape[0]):
        narr[i] = arr[i, :] / vecnorm(arr[i, :])

    return narr


@numba.njit
def cross(a, b):
    '''
    Compute cross product between two vectors
    Uses: Numpy's latest cross product routine
    Arguments:
        a,b                                 1D arrays

    Output:
        Cross product 1D array
    '''

    out = np.zeros(3, dtype=np.float64)

    out[0] = a[1] * b[2]
    tmp = a[2] * b[1]
    out[0] -= tmp

    out[1] = a[2] * b[0]
    tmp = a[0] * b[2]
    out[1] -= tmp

    out[2] = a[0] * b[1]
    tmp = a[1] * b[0]
    out[2] -= tmp

    return out


def get_normals(point_tags, all_tags, coords, trigs):
    '''
    Given a patch defined by the node IDs given by point_tags
    compute the surface normal of the patch. This is done by
    completing the triangles given by the set of point_tags,
    then computing normal of each vertex weighted by triangle
    area. Finally the set of normals across vertices defined
    in the patch is averaged to yield the patch normal.

    Arguments:
        point_tags          Set of Node IDS defining the patch
                            to compute the normal for
        all_tags            The set of all Node IDs belonging
                            to the mesh
        coords              The coordinates of all the vertices
                            belonging to the mesh
        trigs               Array describing triangle faces

    Output:
        A 1x3 vector of the patch normal
    '''

    t_arr = get_relevant_triangles(point_tags, trigs)
    rel_ind = np.where(t_arr > 0)
    rel_trig = trigs[rel_ind[0], :]

    u_val = np.unique(rel_trig)
    u_ind = np.arange(0, u_val.shape[0])
    sort_map = {v: i for v, i in zip(u_val, u_ind)}
    map_func = np.vectorize(lambda x: sort_map[x])

    mapped_trigs = map_func(rel_trig)
    rel_verts = np.where(np.isin(all_tags, u_val))
    rel_verts_coords = coords[rel_verts, :][0]

    norm_array = get_vert_norms(mapped_trigs, rel_verts_coords)
    return norm_array.mean(axis=0)


def ray_interception(pn, pf, coords, trigs, epsilon=1e-6):
    '''
    Compute interception point and distance of ray to mesh defined
    by a set of vertex coordinates and triangles. Yields minimum coordinate
    of ray.

    Algorithm vectorized and
    adapted from http://geomalgorithms.com/a06-_intersect-2.html

    Arguments:
        pn, pf              Points to define ray (near, far)
        coords              Array of vertex coordinates defining mesh
        trigs               Array of vertex IDs defining mesh triangles

    Returns:
        p_I                 Minimum Point of intersection
        ray_len             Length of line from pn to p_I
        min_trig            Triangle ID that contains the shortest
                            length intersection
    '''

    # Get coordinates for triangles
    V0 = coords[trigs[:, 0], :]
    V1 = coords[trigs[:, 1], :]
    V2 = coords[trigs[:, 2], :]

    # Calculate triangle plane
    u = V1 - V0
    v = V2 - V0
    n = np.cross(u, v)

    # Remove all degenerate triangles
    valid_verts = np.where(vecnorm(n, axis=1) > epsilon)
    u = u[valid_verts]
    v = v[valid_verts]
    n = n[valid_verts]

    # Check if ray is in plane w/triangle and if ray is moving toward triangle
    r_denom = (n * (pf - pn)).sum(axis=1)
    r_numer = (n * (V0[valid_verts] - pn)).sum(axis=1)
    r = r_numer / r_denom
    ray_valid = np.where((np.abs(r_denom) > epsilon) & (r > 0))
    u = u[ray_valid]
    v = v[ray_valid]
    n = n[ray_valid]

    # Solve for intersection point of ray to plane
    i_p = pn - (r[:, np.newaxis][ray_valid] * (pn - pf))

    # Check whether the point of intersection lies within the triangle
    w = i_p - V0[valid_verts][ray_valid]
    s, t = compute_parameteric_coordinates(u, v, w)
    s_conditional = ((s > 0) & (s < 1))
    t_conditional = ((t > 0) & (t < 1))
    within_trig = np.where(s_conditional & t_conditional & ((s + t) < 1))

    # Get minimizing triangle identity if a triangle is identified
    if len(within_trig[0]) > 0:
        argmin_r = np.argmin(r[ray_valid][within_trig])
        trig_ids = np.arange(
            0, trigs.shape[0])[valid_verts][ray_valid][within_trig]
        min_trig = trig_ids[argmin_r]

        # Compute distance from ray origin to triangle
        p_I = i_p[within_trig][argmin_r]
        ray_len = vecnorm(p_I - pn)

        # Return point of intersection point, ray length, and triangle with minimum
        return (p_I, ray_len, min_trig)
    else:
        return (None, None, None)


def compute_parameteric_coordinates(u, v, w):
    '''
    Given two coordinate axes (u,v) find (s,t), given that:

                        w = su + tv

    Arguments:
        (u,v)       Parameteric edges
        w           Vector from ray to a point on the plane

    Outputs:
        s           Parameteric coordinate for edge u
        t           Parameteric coordinate for edge v
    '''

    uu = (u * u).sum(axis=1)
    uv = (u * v).sum(axis=1)
    vv = (v * v).sum(axis=1)
    wu = (w * u).sum(axis=1)
    wv = (w * v).sum(axis=1)

    D = (uv * uv) - (uu * vv)
    s = ((uv * wv) - (vv * wu)) / D
    t = ((uv * wu) - (uu * wv)) / D

    return (s, t)


@numba.njit
def get_vert_norms(trigs, coords):
    '''
    Compute vertex normals using cumulative normalization trick
    Arguments:
        trigs          Array of triangles with normalized values
                       range(1,size(unique(trigs)))
        coords         Array of coordinates
                       (vals in trigs correspond to inds in coords)
    Output:
        norm_arr       Array of norm vectors
    '''

    cnorm_arr = np.zeros((coords.shape[0], 3), dtype=np.float64)
    for i in np.arange(0, trigs.shape[0]):

        iv1 = trigs[i, 0]
        iv2 = trigs[i, 1]
        iv3 = trigs[i, 2]

        v1 = coords[iv1, :]
        v2 = coords[iv2, :]
        v3 = coords[iv3, :]

        c = cross(v2 - v1, v3 - v1)

        cnorm_arr[iv1, :] += c
        cnorm_arr[iv2, :] += c
        cnorm_arr[iv3, :] += c

    norm_arr = unitize_arr(cnorm_arr)
    return norm_arr


@numba.njit(parallel=True)
def get_subset_triangles(verts, triangles):
    '''
    From an array of vertices and triangles,
    get the triangles that contain all vertices
    Arguments:
        verts          1-D array of vertex IDs
        triangles      (Nx3) array of triangles where each column is a vertex

    Output:
        t_arr          Nx1 Boolean array where indices correspond to
                       triangles. True if all 3 vertices of triangle
                       found in verts
    '''

    t_arr = np.zeros((triangles.shape[0]), dtype=np.int64)

    for t in numba.prange(0, triangles.shape[0]):
        for c in np.arange(0, 3):
            for v in verts:

                if triangles[t][c] == v:
                    t_arr[t] += 1
                    break

        if t_arr[t] == 3:
            t_arr[t] = 1
        else:
            t_arr[t] = 0

    return t_arr
