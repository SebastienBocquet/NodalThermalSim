import copy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
import numpy as np
from NodalThermalSim.Solver import HALF_STENCIL
from NodalThermalSim.Physics import BoundaryConditionFlux, BoundaryConditionDirichlet


X = 0
Y = 1
Z = 2


class GridBase():

    GHOST_INDEX = {'left': 0, 'right': -1}
    BOUNDARY_VAL_INDEX = {'left': 1, 'right': -2}
    FIRST_PHYS_VAL_INDEX = {'left': 2, 'right': -3}

    def __init__(self, resolution, y0, delta_x, dx=-1):
        """

        Args:
            name: name of the component
            resolution: number of mesh points
            y0: initial value for y
            delta_x: length of the mesh
            dx: length of mesh cells. If negative or null,
            dx is determined from resolution. If stricly positive,
            resolution is determined from dx.
        """
        if dx > 0:
            self.dx = dx
            self.resolution = (int)(delta_x / dx + 1)
        else:
            self.resolution = resolution
            self.dx = delta_x / (self.resolution - 1)
        assert(self.resolution > 1)
        x0 = 0.
        self.x = np.linspace(x0 - HALF_STENCIL * self.dx,
                             (self.resolution - 1 + HALF_STENCIL) * self.dx,
                             num=self.resolution + 2 * HALF_STENCIL)
        self.val = np.zeros((self.resolution + 2 * HALF_STENCIL))
        self.val[HALF_STENCIL:self.resolution + HALF_STENCIL] = y0
        self.boundary = {'left': BoundaryConditionDirichlet(), 'right': BoundaryConditionDirichlet()}
        self.neighbours = {'left': None, 'right': None}
        self.neighbour_faces = {'left': 'right', 'right': 'left'}
        self.ghost_target_val = {'left': 0., 'right': 0.}

    def set_boundary(self, boundary):
        self.boundary = boundary
        assert self.boundary.keys() == self.GHOST_INDEX.keys()

    def set_neighbours(self, neighbours, faces={'left': 'right', 'right': 'left'}):
       self.neighbours = neighbours
       self.neighbour_faces = faces
       assert self.neighbours.keys() == self.neighbour_faces.keys()

    def setGhostValue(self, face, val):
        # print('face, ghost val', face, val)
        self.val[self.GHOST_INDEX[face]] = val

    def setBoundaryValue(self, face, boundary_value):
        self.val[self.BOUNDARY_VAL_INDEX[face]] = boundary_value

    def get_ghost_value(self, loc):
       return self.val[self.GHOST_INDEX[loc]]

    def get_boundary_value(self, loc):
       return self.val[self.BOUNDARY_VAL_INDEX[loc]]

    def get_first_phys_value(self, loc):
       return self.val[self.FIRST_PHYS_VAL_INDEX[loc]]

    def get_physics_x(self):
       return self.x[HALF_STENCIL:self.resolution+HALF_STENCIL]

    def get_physics_val(self):
       return self.val[HALF_STENCIL:self.resolution+HALF_STENCIL]

    # gradient is oriented towards the exterior of the component.
    # TODO rename loc in face
    def get_boundary_gradient(self, loc):
        return (self.get_ghost_value(loc) - self.get_boundary_value(loc)) / self.dx
