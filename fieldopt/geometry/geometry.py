#!/usr/bin/env python
# coding: utf-8

"""General geometry routines for mesh manipulation
"""


import numpy as np
from scipy.linalg import lstsq
import numba


def skew(vector):
    """
    This function returns a numpy array with the skew symmetric cross product
    matrix for vector. The skew symmetric cross product matrix is defined
    such that:

    `np.cross(a, b) = np.dot(skew(a), b)`

    Arguments:
        vector (ndarray): An array like vector to create the skew
            symmetric cross product matrix for

    Returns:
        skew (ndarray): A numpy array of the skew symmetric cross
            product vector

    References:
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
    Apply affine transformation A to vector set x.

    Arguments:
        A (ndarray): (4,4) Affine matrix
        x (ndarray): (3,N) Set of 3D coordinate vectors

    Returns:
        b (ndarray): Transformed coordinates :math: `Ax`
    '''

    h = np.c_[x, np.ones((x.shape[0], 1))]
    Ah = A @ h.T
    return Ah.T[:, :3]


def rotate_vec2vec(v1, v2):
    '''
    Rotate vector v1 onto v2 and return the transformation matrix R that
    achieves this

    Compute transformation matrix :math:`R` that rotates a vector
    :math:`v1` onto :math:`v2`

    Arguments:
        v1 (ndarray): (3,) starting vector
        v2 (ndarray): (3,) final vector to rotate to

    Returns:
        R (ndarray): (3,3) rotation matrix
    '''

    n = np.cross(v1, v2)
    sinv = np.linalg.norm(n)
    cosv = np.dot(v1, v2)
    R = (np.eye(3) + skew(n) + np.matmul(skew(n), skew(n)) * (1 - cosv) /
         (sinv**2))
    return R


def quad_fit(X, b):
    '''
    Perform quadratic surface fitting of form:

    :math:`f(x,y) = a + bx + cy + dxy + ex^2 + fy^2`

    By finding the least squares solution of :math:`Ax = b`

    Arguments:
        X (ndarray): (P, 2) P set of :math:`(x,y)` points
        b (ndarray): (P,) set of :math:`f(x,y)` points
    '''

    # Formulate the linear problem
    A = np.c_[np.ones((X.shape[0], 1)), X[:, :2],
              np.prod(X[:, :2], axis=1), X[:, :2]**2]

    C, _, _, _ = lstsq(A, b)
    return C


def compute_principal_dir(x, y, C):
    '''
    Compute the principal direction of a quadratic surface :math:`S` of form:

    :math:`S: f(x,y) = a + bx + cy + dxy + ex^2 + fy^2`

    Using the second fundamental form basis matrix eigendecomposition method

    Arguments:
        x (float): :math:`x` coordinate on parameterized surface :math:`S`
        y (float): :math:`y` coordinate on parameterized surface :math:`S`
        C (ndarray): (6,) surface coefficients

    Returns:
        V (ndarray): (2,2) Principal component column vectors
        n (ndarray): (3,) normal to surface S at point (x,y)
    '''

    # Compute partial first and second derivatives
    r_x = np.array([1, 0, 2 * C[4] * x + C[1] + C[3] * y])
    r_y = np.array([0, 1, 2 * C[5] * y + C[2] + C[3] * x])
    r_xx = np.array([0, 0, 2 * C[4]])
    r_yy = np.array([0, 0, 2 * C[5]])
    r_xy = np.array([0, 0, C[3]])

    # Compute surface point normal
    r_x_cross_y = np.cross(r_x, r_y)
    n = r_x_cross_y / np.linalg.norm(r_x_cross_y)

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


def interpolate_angle(u, v, t, l=90.0):  # noqa: E741
    '''
    Interpolate between two orthogonal vectors :math:`u` and :math:`v`

    Arguments:
        u (ndarray): (3,) vector
        v (ndarray): (3,) vector orthogonal to :math:`u`
        t (float): Interpolation value
        l (float): Period of rotation

    Returns:
        ori (ndarray): (3,) vector in between :math:`u` and :math:`v`
    '''

    theta = (t / l) * (np.pi / 2)
    p = np.r_[u * np.cos(theta) + v * np.sin(theta), 0]
    return p


def quadratic_surf(x, y, C):
    '''
    Project parameteric coordinates :math:`(x,y)` onto a quadratic surface
    :math:`S` defined as:

    :math:`f(x,y) = a + bx + cy + dxy + ex^2 + fy^2`

    Arguments:
        x (float): :math:`x` coordinate
        y (float): :math:`y` coordinate
        C (ndarray): (6,) surface coefficients

    Returns:
        :math:`f(x,y)` scalar value
    '''

    return (C[0] + C[1] * x + C[2] * y + C[3] * x * y + C[4] * x * x +
            C[5] * y * y)


def quadratic_surf_position(x, y, C):
    '''
    For some mesh-based surface :math:`S`,
    define a parameterization using a quadratic fit:

    :math:`f(x,y) = a + bx + cy + dxy + ex^2 + fy^2`

    Compute the mapping from :math:`(x,y) \\to (x,y,f(x,y))`

    Arguments:
        x (float): :math:`x` coordinate
        y (float): :math:`y` coordinate
        C (ndarray): (6,) surface coefficients

    Returns:
        (3,) :math:`(x,y,f(x,y))` vector
    '''

    # Compute approximate surface at (x,y)
    z = quadratic_surf(x, y, C)

    # Form input vector
    v = np.array([x, y, z], dtype=np.float64)

    return v


def quadratic_surf_rotation(x, y, t, C):
    '''
    Construct an orientation vector on a quadratic surface defined by
    coefficients :math:`C`

    For some mesh-based surface, define a least squares quadratic
    surface parameterization :math:`S`:

    :math:`f(x,y) = a + bx + cy + dxy + ex^2 + fy^2`

    :math:`(x,y)` are therefore parameteric coordinates that are projected
    to the surface :math:`S`. Following projection a local orientation is
    defined by interpolating the principal directions
    :math:`(u,v)` by :math:`t`.

    Arguments:
        x (float): :math:`x` coordinate
        y (float): :math:`y` coordinate
        t (float): Interpolation amount between principal directions
            :math:`(u,v)`. Period of :math:`T=90` is used for
            interpolation.
        C (ndarray): (6,) surface coefficients

    Returns:
        (3,) :math:`(x,y,z)` direction vector
        (3,) normal vector
    '''

    v1, v2, n = compute_principal_dir(x, y, C)
    p = interpolate_angle(v1[:2], v2[:2], t)

    z = np.array([0, 0, 1], dtype=np.float64)
    R = rotate_vec2vec(z, n)
    pp = np.matmul(R, p)

    return pp, n


def define_coil_orientation(loc, rot, n):
    '''
    Construct the coil orientation matrix (matsimnibs) to be used by simnibs

    Arguments:
        loc (ndarray): (3,) Centre coordinates of coil
        rot (ndarray): Orientation vector of coil
        n (ndarray): Coil normal vector

    Returns:
        matsimnibs (ndarray): (4,4) matsimnibs matrix with column vectors
            [x, y, z, t]
    '''

    y = rot / np.linalg.norm(rot)
    z = n / np.linalg.norm(n)
    x = np.cross(y, z)
    c = loc

    matsimnibs = np.zeros((4, 4), dtype=np.float64)
    matsimnibs[:3, 0] = x
    matsimnibs[:3, 1] = y
    matsimnibs[:3, 2] = z
    matsimnibs[:3, 3] = c
    matsimnibs[3, 3] = 1

    return matsimnibs


@numba.njit
def unitize_arr(arr):
    '''
    Normalize array row-wise

    Arguments:
        arr (ndarray): (n,p) array to normalize

    Returns:
        normed (ndarray): (n,p) array where each row is unit length
    '''

    # TODO: Generalize this to p dimensions
    narr = np.zeros((arr.shape[0], 3), dtype=np.float64)

    for i in np.arange(0, arr.shape[0]):
        narr[i] = arr[i, :] / np.linalg.norm(arr[i, :])

    return narr


def compute_parameteric_coordinates(u, v, w):
    '''
    Given two coordinate axes :math:`(u,v)` find :math:`(s,t)`, given that:

    :math:`w = su + tv`

    Arguments:
        u (ndarray): (3,) Axis of parameteric coordinate system
        v (ndarray): (3,) Secondary axis of parameteric coordinate system
        w (ndarray): (3,) Point lying on plane spanned by :math:`(u,v)`

    Returns:
        s (float): Parameteric coordinate for axis spanned by :math:`u`
        t (float): Parameteric coordinate for axis spanned by :math:`v`
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
