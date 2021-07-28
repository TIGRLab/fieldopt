'''
Data class for operating with simnibs.msh.mesh_io.Msh
objects more concisely for running TMS simulation experiments
'''

import numpy as np
from simnibs.msh.mesh_io import read_msh


# TODO: Make dataclass
class HeadModel:
    def __init__(self, mesh, head_entity=(2, 5)):

        if isinstance(mesh, str):
            self.mesh = read_msh(mesh)
        else:
            self.mesh = mesh

        self.mesh.fix_surface_labels()
        head = self.mesh.crop_mesh(elm_type=2).crop_mesh(tags=1005)
        self.nodes = head.nodes.node_number
        self.coords = head.nodes.node_coord
        self.trigs = head.elm.node_number_list[:, :3].copy()

    def get_tet_ids(self, tag):
        return np.where(self.mesh.elm.tag1 == tag)

    def intercept(self, p0, p1, entity):
        '''
        Compute interception distance of a ray defined by (p0, p1) to
        a mesh entity

        Arguments:
            p0                  Ray starting point
            p1                  Ray end point
            entity              (tag, elm) entity
                                to compute intersection with
        '''

        tag, elm = entity
        m = self.mesh.crop_msh(tag).crop_msh(elm)
        return m.intercept_ray(p0, p1)
