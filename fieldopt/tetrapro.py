# !/usr/bin/env python
# coding: utf-8
"""
Fieldopt tetrahedral projection module for mapping
volumetric data onto a tetrahedral volumetric mesh model
"""

import numba
import numpy as np


@numba.njit
def map_nodes(x, prop_array):
    '''
    Convenience function to remap a value according to a properties array.
    A properties array is an (nx3) numpy array that stores
    the following information for the :math:`i`th gmsh element

    - [1] - minimum element node number
    - [2] - maximum element node number
    - [3] - number of element nodes

    Arguments:
        x (ndarray): Node array to remap
        prop_array (ndarray): (K, 3) properties array

    Returns:
        Remapped x `ndarray` according to `prop_array`
    '''

    out = np.zeros_like(x, dtype=np.int64)
    for i in np.arange(x.shape[0]):
        for j in np.arange(0, x.shape[1]):
            for k in np.arange(0, prop_array.shape[0]):

                if (x[i, j] >= prop_array[k, 0]) & (x[i, j] <= prop_array[k,
                                                                          1]):
                    out[i, j] = x[i, j] - prop_array[k, 0] + np.sum(
                        prop_array[:k, 2])

    return out


@numba.njit
def homogenous_transform(coords, A):
    '''
    Transform into homogenous coordinates and apply an affine map

    Arguments:
        coords (ndarray): (1,3) array to transform
        A (ndarray): (4,4) Affine map to apply
    '''

    # Simpler implementation
    coords = np.dot(A[:3, :3], coords.T)
    coords += A[:3, 3:4]
    return coords.T


@numba.njit
def meshgrid(x, y, z):
    '''
    Create a coordinate array for all combinations of points in `x,y,z`

    Arguments:
        x (ndarray): (N,) x-coordinate array
        y (ndarray): (N,) y-coordinate array
        z (ndarray): (N,) z-coordinate array

    Returns:
        (3,P) matrix of all permutations of `x,y,z`
    '''
    # Create output array of all possible combinations
    mg = np.zeros((3, x.size * y.size * z.size), np.int32)

    # For each item in x
    counter = 0
    for i in np.arange(0, x.size):
        for j in np.arange(0, y.size):
            for k in np.arange(0, z.size):

                mg[0, counter] = x[i]
                mg[1, counter] = y[j]
                mg[2, counter] = z[k]
                counter += 1
    return mg


# TODO: Remove 1mm voxel size assumption
@numba.njit
def aabb_voxels(coords):
    '''
    Identify voxels within an axis-aligned boundary box that contain
    coordinates in `coords`

    Arguments:
        coords (ndarray): (4,3) array containing tetrahedral coordinates
            in voxel space

    Returns:
        Voxel (i,j,k) coordinates containing a coordinate in `coords`

    Notes:
        Assumes voxels are isotropic 1x1x1!
    '''

    # Pre-allocate and store bounds
    min_vox = np.zeros((3), np.int32)
    max_vox = np.zeros((3), np.int32)

    # Get min, max then floor and ceil respectively
    for i in np.arange(0, 3):
        min_vox[i] = np.min(coords[:, i])
        max_vox[i] = np.max(coords[:, i])
    min_vox = np.floor(min_vox)
    max_vox = np.floor(max_vox)

    # Get voxel set
    x_range = np.arange(min_vox[0], max_vox[0] + 1, 1, np.int32)
    y_range = np.arange(min_vox[1], max_vox[1] + 1, 1, np.int32)
    z_range = np.arange(min_vox[2], max_vox[2] + 1, 1, np.int32)
    vox_arr = meshgrid(x_range, y_range, z_range)

    return vox_arr


@numba.njit
def uniform_tet(coords):
    '''
    Generate a sample within a tetrahedron from a uniform distribution

    Argument:
        coords (ndarray): (4,3) tetrahedral vertex array

    Returns:
        A random point inside the tetrahedral volume defined by `coords`

    References:
        http://vcg.isti.cnr.it/jgt/tetra.htm
    '''

    s = np.random.uniform(0, 1)
    t = np.random.uniform(0, 1)
    u = np.random.uniform(0, 1)

    # First cut
    if (s + t > 1):
        s = 1.0 - s
        t = 1.0 - t

    # Second set of cuts
    if (t + u > 1):
        tmp = u
        u = 1.0 - s - t
        t = 1.0 - tmp
    elif (s + t + u > 1):
        tmp = u
        u = s + t + u - 1
        s = 1 - t - tmp

    a = 1 - s - t - u

    return a * coords[0] + s * coords[1] + t * coords[2] + u * coords[3]


@numba.njit
def point_in_vox(point, midpoint, voxdim=1):
    '''
    Check whether `point` is inside a voxel defined by its
    `midpoint` and `voxdim`

    Arguments:
        point (ndarray): (3,) point to check
        midpoint (ndarray): (3,) voxel midpoint
        voxdim (float): voxel width, assuming isotropic

    Output:
        True if point in voxel bounds
    '''

    # Shift midpoint upwards by half a voxel (left,top,back --> centre of cube)
    halfvox = voxdim / 2.
    midpoint = midpoint + halfvox

    # Checks
    if (point[0] < midpoint[0] - halfvox) or (point[0] >
                                              midpoint[0] + halfvox):
        return False
    elif (point[1] < midpoint[1] - halfvox) or (point[1] >
                                                midpoint[1] + halfvox):
        return False
    elif (point[2] < midpoint[2] - halfvox) or (point[2] >
                                                midpoint[2] + halfvox):
        return False
    else:
        return True


@numba.njit
def estimate_partial_parcel(coord, vox, parcels, out, n_iter=300):
    '''
    Estimate parcellation labels composition of a tetrahedron

    Arguments:
        coord (ndarray): (4,3) tetrahedron vertices
        vox (ndarray): (n,3) voxel coordinates
        parcels (ndarray): (n,1) voxel parcel labels
        out (ndarray): output array
        iter (int): number of Monte-Carlo sampling interations

    Returns:
        (P,) `ndarray` of how much of each parcel is in a tetrahedron
    '''

    # Check degenerate case
    if np.unique(parcels).shape[0] == 1:
        out[parcels[0]] = 1

    # Shift tetrahedron to origin
    t = coord[0]
    coord = coord - t

    # Perform fixed monte carlo sampling
    for i in np.arange(0, n_iter):

        resample = True
        p = uniform_tet(coord)
        for j in np.arange(0, vox.shape[1]):

            # If point is in voxel, then move on
            if point_in_vox(p + t, vox[:, j]):
                resample = False
                out[parcels[j]] += 1
                break

        if resample:
            i -= 1


@numba.njit(parallel=True)
def tetrahedral_parcel_projection(node_list,
                                  coord_arr,
                                  ribbon,
                                  affine,
                                  n_iter=300):
    '''
    Perform tetrahedral projection for parcellation inputs

    Arguments:
        node_list (ndarray): (T, 4) List of tetrahedral vertex IDs
        coord_arr (ndarray): (N, 3) Vertex coordinate list
        ribbon (ndarray): (X,Y,Z) MRI parcellation volume array
        affine (ndarray): Affine transformation matrix associated with `ribbon`
        n_iter (int): Number of monte-carlo iterations to estimate proportion
            of volume

    Returns:
        (T,P) array containing parcel compositions for each tetrahedron
    '''

    # Compute inverse affine
    inv_affine = np.linalg.inv(affine)

    # Loop tetrahedrons
    num_elem = node_list.shape[0]

    # Total number of parcels
    num_parc = int(ribbon.max())

    # make output array
    out_arr = np.zeros((num_elem, num_parc + 1), dtype=np.float64)

    linear_coord = coord_arr.flatten()

    for i in numba.prange(0, num_elem):

        # Get coordinates for nodes
        t_coord = np.zeros((4, 3), dtype=np.float64)

        # Set up arrays for indexing
        i_1 = get_vertex_range(node_list, i, 0, 3)
        i_2 = get_vertex_range(node_list, i, 1, 3)
        i_3 = get_vertex_range(node_list, i, 2, 3)
        i_4 = get_vertex_range(node_list, i, 3, 3)

        # Linear indexing
        t_coord[0, :] = linear_coord[i_1]
        t_coord[1, :] = linear_coord[i_2]
        t_coord[2, :] = linear_coord[i_3]
        t_coord[3, :] = linear_coord[i_4]

        # Step 1: Transform coordinates to MR space
        t_coord[0:1, :] = homogenous_transform(t_coord[0:1, :], inv_affine)
        t_coord[1:2, :] = homogenous_transform(t_coord[1:2, :], inv_affine)
        t_coord[2:3, :] = homogenous_transform(t_coord[2:3, :], inv_affine)
        t_coord[3:4, :] = homogenous_transform(t_coord[3:4, :], inv_affine)

        # Step 2: Perform axis-aligned boundary box finding
        vox_arr = aabb_voxels(t_coord)

        # Step 3: Get parcel values associated with voxels
        parcels = np.zeros((vox_arr.shape[1] + 1), np.int32)
        for j in np.arange(vox_arr.shape[1]):
            parcels[j] = ribbon[vox_arr[0, j], vox_arr[1, j], vox_arr[2, j]]

        # Step 4: Estimate partial parcels
        estimate_partial_parcel(t_coord, vox_arr, parcels, out_arr[i, :],
                                n_iter)

    return out_arr / n_iter


# TODO: Make better description/function name
@numba.njit
def get_vertex_range(arr, i, j, k):
    '''
    Wrapper for numba to be able to use tuple-based indexing with Python 3.6+.

    Given an array arr, an index base i, and a stride of k.
    Generate an array which starts at k*arr[i,j] and goes to (k*arr[i,j])+k

    Arguments:
        arr (ndarray): Array to index
        i (int): row to select
        j (int): column to select
        k (int): end point of range

    Returns:
        Range from `k*a[i,j]` to `k*[a[i,j]]+k`
    '''

    start = k * arr[i, j]
    end = start + k
    indices = np.arange(start, end)
    return indices


@numba.njit(parallel=True)
def tetrahedral_weight_projection(node_list,
                                  coord_arr,
                                  ribbon,
                                  affine,
                                  n_iter=300):
    '''
    Perform tetrahedral projection for weighted inputs

    Arguments:
        node_list (ndarray): (T, 4) List of tetrahedral vertex IDs
        coord_arr (ndarray): (N, 3) Vertex coordinate list
        ribbon (ndarray): (X,Y,Z) Continously valued volume array
        affine (ndarray): Affine transformation matrix associated with `ribbon`
        n_iter (int): Number of monte-carlo iterations to estimate proportion
            of volume

    Returns:
        (T,) array containing `ribbon` values resampled to
            tetrahedral volume mesh
    '''
    # Compute inverse affine
    inv_affine = np.linalg.inv(affine)

    # Loop tetrahedrons
    num_elem = node_list.shape[0]

    # make output array
    out_arr = np.zeros((num_elem), dtype=np.float64)

    # Flatten the coordinate array for fast linear indexing
    linear_coord = coord_arr.flatten()

    for i in numba.prange(0, num_elem):

        # Get coordinates for nodes
        t_coord = np.zeros((4, 3), dtype=np.float64)

        # Set up arrays for indexing
        i_1 = get_vertex_range(node_list, i, 0, 3)
        i_2 = get_vertex_range(node_list, i, 1, 3)
        i_3 = get_vertex_range(node_list, i, 2, 3)
        i_4 = get_vertex_range(node_list, i, 3, 3)

        # Linear indexing
        t_coord[0, :] = linear_coord[i_1]
        t_coord[1, :] = linear_coord[i_2]
        t_coord[2, :] = linear_coord[i_3]
        t_coord[3, :] = linear_coord[i_4]

        # Step 1: Transform coordinates to MR space
        t_coord[0:1, :] = homogenous_transform(t_coord[0:1, :], inv_affine)
        t_coord[1:2, :] = homogenous_transform(t_coord[1:2, :], inv_affine)
        t_coord[2:3, :] = homogenous_transform(t_coord[2:3, :], inv_affine)
        t_coord[3:4, :] = homogenous_transform(t_coord[3:4, :], inv_affine)

        # Step 2: Perform axis-aligned boundary box finding
        vox_arr = aabb_voxels(t_coord)

        # Step 3: Iterate through candidate voxels and pull weights
        weights = np.zeros((vox_arr.shape[1] + 1), np.float64)
        for j in np.arange(vox_arr.shape[1]):
            weights[j] = ribbon[vox_arr[0, j], vox_arr[1, j], vox_arr[2, j]]

        # Step 4: Estimate weighted score
        estimate_partial_weights(t_coord, vox_arr, weights, out_arr[i:i + 1],
                                 n_iter)

    return out_arr / n_iter


@numba.njit
def estimate_partial_weights(coord, vox, weights, out, n_iter=300):
    '''
    Estimate resampled value of a tetrahedron from voxels

    Arguments:
        coord (ndarray): (4,3) tetrahedron vertices
        vox (ndarray): (n,3) voxel coordinates
        weights (ndarray): (n,1) voxel values
        out (ndarray): output array to write to
        iter (int): number of Monte-Carlo sampling interations

    Returns:
        Scalar value of tetrahedron from resampling from voxel weights
    '''

    # No degenerate case exists
    # Shift tetrahedron to origin
    t = coord[0]
    coord = coord - t

    # Perform fixed monte carlo sampling
    for i in np.arange(0, n_iter):

        resample = True
        p = uniform_tet(coord)
        for j in np.arange(0, vox.shape[1]):

            # If point is in voxel, then move on
            if point_in_vox(p + t, vox[:, j]):
                resample = False
                out += weights[j]
                break

        if resample:
            i -= 1
