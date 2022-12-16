import copy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
import numpy as np
from Solver import HALF_STENCIL


X = 0
Y = 1
Z = 2

class BoundaryConditionFlux:

    def __init__(self, type_='heatFlux', flux=0.):
        self.type = type_
        self.flux = flux

    def compute(self, face, neigh, neighbour_face, boundary_value, dx, thermal_conductivity):
        if self.type == 'heatFlux':
            gradient = self.flux / thermal_conductivity
            ghost_val = boundary_value - dx * gradient
            return ghost_val,
        else:
            raise ValueError


class BoundaryConditionDirichlet:

    def __init__(self, type='conservative'):
        self.type = type

    def compute(self, face, neigh, neighbour_face, boundary_value, dx, thermal_conductivity):
            # fill ghost node with the neighbour first physical node value, corrected to impose
            # a gradient that ensures heat flux conservation through component interface.
            ghost_val = neigh.get_grid().get_boundary_value(neighbour_face)
            ghost_target = 0.
            if self.type == 'conservative':
                # TODO log an info that conservative treatment is deactivated
                if neigh.material is not None:
                    gradient_neighbour = (neigh.get_grid().get_boundary_value(neighbour_face) -
                                          neigh.get_grid().get_first_phys_value(neighbour_face)) / neigh.get_grid().dx
                    flux_neighbour = neigh.material.thermal_conductivity * gradient_neighbour
                    flux_target = -flux_neighbour
                    gradient = flux_target / thermal_conductivity
                    ghost_target = boundary_value - dx * gradient
                    error = ghost_val - ghost_target
                    ghost_val += 1. * error
                return ghost_val, ghost_target
            elif self.type == 'non_conservative':
                return ghost_val,
            else:
                raise ValueError


class GridBase(ABC):

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

    @abstractmethod
    def get_boundary_gradient(self, loc):
        return

    def setGhostValue(self, face, val, target=0.):
        self.val[self.GHOST_INDEX[face]] = val
        self.ghost_target_val[face] = target

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


class Grid1D(GridBase):

    """
    Grid1D
    number of ghost node on each side is HALF_STENCIL.
    There are (resolution) physical nodes, plus (2 * HALF_STENCIL) ghost nodes.
    There are (resolution-1) cells in the physical range.
    Normal orientation is: left--->right
    """

    def __init__(
        self,
        thickness,
        y0,
        resolution=10,
        dx=-1,
    ):

        super().__init__(resolution, y0, thickness, dx)
        self.thickness = thickness

    # gradient is oriented towards the exterior of the component.
    # TODO rename loc in face
    def get_boundary_gradient(self, loc):
        # return (self.get_boundary_value(loc) - self.get_first_phys_value(loc)) / self.dx
        return (self.get_ghost_value(loc) - self.get_boundary_value(loc)) / self.dx

    def get_boundary_heat_flux(self, loc, ax='x'):
        # two approaches: either use the local data (boundary and ghost value,
        # relying on the fact that heat flux is conserved through the boundary.
        # or use the neighbour heat flux.
        # return self.get_boundary_gradient(loc, axis) * self.material.thermal_conductivity

        neigh = self.neighbours[loc]
        neighbour_face = self.neighbour_faces[loc]

        if neigh is None:
            assert self.boundary[loc].type == 'heatFlux'
            return -self.boundary[loc].flux
        else:
            if neigh.material is None:
                assert self.boundary[loc].type == 'heatFlux'
                return -self.boundary[loc].flux

        gradient_neighbour = -(neigh.get_grid(ax).get_boundary_gradient(neighbour_face))
        return neigh.material.thermal_conductivity * gradient_neighbour