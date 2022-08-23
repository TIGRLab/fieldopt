"""Gmsh mesh manipulation routines
"""
import numpy as np
import numba
import fieldopt.geometry.geometry as geometry
import gmsh


def load_gmsh_nodes(gmshpath, entity):
    '''
    Load gmsh .msh file and load (nodes, coordinates, parameters)

    Arguments:
        gmshpath (str): Path to .msh file
        entity (tuple): (dim, tag) tuple identifying the entity
            to query

    Returns:
        nodes (ndarray): (N,) Node ID array shifted to 0-based indexing
        coords (ndarray): (N, 3) node coordinate array
        parameters (ndarray): (N, P) node parameter array
    '''

    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, coords, params = gmsh.model.mesh.getNodes(entity[0], entity[1])
    coords = np.array(coords).reshape((len(coords) // 3, 3))
    gmsh.clear()

    return nodes - 1, coords, params


def load_gmsh_elems(gmshpath, entity):
    '''
    Load gmsh .msh file and load (elm_types, ids, node_maps)

    Arguments:
        gmshpath (str): Path to .msh file
        entity (tuple): (dim, tag) tuple identifying the entity
            to query

    Returns:
        elm_types (ndarray): (N,) Element type array
            (3=triangle, 4=tetrahedron)
        ids (ndarray): (N, P) Element ID array shifted to 0-based indexing
        node_map (ndarray): (N, DIM) Vertex IDs for element.
            DIM=3 if triangles. DIM=4 if tetrahedrons.
    '''

    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, elem_ids, node_maps = gmsh.model.mesh.getElements(
        entity[0], entity[1])
    gmsh.clear()

    return nodes, elem_ids[0] - 1, node_maps[0] - 1


def closest_point2surf(p, coords):
    '''
    Get closest point :math:`c` in coords to :math:`p`
    by minimizing the euclidean distance metric

    Arguments:
        p (ndarray): (3,) point
        coords (ndarray): (N, 3) set of coordinates

    Returns:
        (3,) coordinate :math:`c` closest to :math:`p` in `coords`
    '''

    ind = np.argmin(np.linalg.norm(coords - p, axis=1))
    return coords[ind, :]


def get_neighbours(indices, triangles):
    """
    Return neighbouring vertices around `indices`

    Arguments:
        indices (ndarray): (V,) array to compute neighbourhood around
        triangles (ndarray): (P,3) array of triangles

    Returns:
        neighbours (ndarray): (K,) array of neighbouring vertices including
            the nodes in indices
    """

    result = get_relevant_triangles(indices, triangles)
    connected_trigs = np.where(result)
    return np.unique(triangles[connected_trigs, :].flatten())


def get_ring(index, triangles, n_neighbours=2):
    """
    Return an N-ring of nodes around `index`

    Arguments:
        index (int): Index of central vertex
        triangles (ndarray): (P,3) array of triangle vertices

    Returns:
        indices (ndarray): (K,) integer array containing node indices belonging
            to n-ring around `index`
    """

    query_inds = [index]

    for i in range(n_neighbours):
        query_inds = get_neighbours(query_inds, triangles)

    return query_inds


def get_relevant_triangles(verts, triangles):
    '''
    Get set of all triangles in `triangles`
    that contain at least 1 vertex in `verts`.

    Arguments:
        verts (ndarray): (N,) array of vertex IDs
        triangles (ndarray): (P,3) array of triangle vertices

    Returns:
        t_arr (ndarray): (P,) boolean array. True if triangle contains at least
            one vertex from `verts
    '''

    t_arr = np.zeros((triangles.shape[0]), dtype=np.int64)
    t_arr += np.isin(triangles[:, 0], verts)
    t_arr += np.isin(triangles[:, 1], verts)
    t_arr += np.isin(triangles[:, 2], verts)
    return np.clip(t_arr, 0, 1)


def get_normals(point_tags, all_tags, coords, trigs):
    '''
    Get averaged local normal for a contiguous subset of a surface mesh

    Arguments:
        point_tags (ndarray): (K,) Subset of contiguous Node ids
            to use for computing local normal
        all_tags (ndarray): (N,) The set of all Node ids belonging
            to the mesh
        coords (ndarray): (N,3) Coordinates of nodes in `all_tags`
        trigs (ndarray): (P,3) Triangle face array

    Returns:
        (3,) normal vector of patch defined by `point_tags`
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

    norm_array = vertex_normals(mapped_trigs, rel_verts_coords)
    return norm_array.mean(axis=0)


def ray_interception(pn, pf, coords, trigs, epsilon=1e-6):
    '''
    Compute interception point and distance of ray to mesh.
    Only the first collision is returned.

    Arguments:
        pn (ndarray): (3,) starting point of ray
        pf (ndarray): (3,) far point of ray
        coords (ndarray): (N,3) Vertex coordinate array
        trigs (ndarray): (P,3) Triangle face array

    Returns:
        p_I (ndarray): (3,) Intersection point
        ray_len (float): Euclidean length of line from `pn` to `p_I`
        min_trig (int): Index of triangle that first collided with ray

    References:
        http://geomalgorithms.com/a06-_intersect-2.html
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
    valid_verts = np.where(np.linalg.norm(n, axis=1) > epsilon)
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
    s, t = geometry.compute_parameteric_coordinates(u, v, w)
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
        ray_len = np.linalg.norm(p_I - pn)

        return (p_I, ray_len, min_trig)
    else:
        return (None, None, None)


@numba.njit
def vertex_normals(trigs, coords):
    '''
    Compute vertex normals

    Arguments:
        trigs (ndarray): (P,3) Normalized triangle face array (see note)
        coords (ndarray): (N,3) Array of vertex coordinates

    Output:
        (K,3) vertex normals

    Note:
        `trigs` must be normalized such that `max(trigs) = N-1`. So
        that each element value in `trigs` can be used to directly
        index a row in `coords`
    '''

    cnorm_arr = np.zeros((coords.shape[0], 3), dtype=np.float64)
    for i in np.arange(0, trigs.shape[0]):

        iv1 = trigs[i, 0]
        iv2 = trigs[i, 1]
        iv3 = trigs[i, 2]

        v1 = coords[iv1, :]
        v2 = coords[iv2, :]
        v3 = coords[iv3, :]

        c = np.cross(v2 - v1, v3 - v1)

        cnorm_arr[iv1, :] += c
        cnorm_arr[iv2, :] += c
        cnorm_arr[iv3, :] += c

    norm_arr = geometry.unitize_arr(cnorm_arr)
    return norm_arr


@numba.njit(parallel=True)
def get_subset_triangles(verts, triangles):
    '''
    Get all triangles in `triangles` that have all their vertices
    listed in `verts`.

    Arguments:
        verts (ndarray): (N,) array of vertex IDs
        triangles (ndarray): (P,3) triangle face array

    Output:
        (P,) boolean array indicating triangles where all vertices
            are found in `verts`
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
