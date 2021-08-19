import numpy as np
import numba
import fieldopt.geometry.geometry as geometry
import gmsh


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


def closest_point2surf(p, coords):
    '''
    Get closest point c in coords to p
    '''

    ind = np.argmin(np.linalg.norm(coords - p, axis=1))
    return coords[ind, :]


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

    norm_array = vertex_normals(mapped_trigs, rel_verts_coords)
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

        c = np.cross(v2 - v1, v3 - v1)

        cnorm_arr[iv1, :] += c
        cnorm_arr[iv2, :] += c
        cnorm_arr[iv3, :] += c

    norm_arr = geometry.unitize_arr(cnorm_arr)
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
