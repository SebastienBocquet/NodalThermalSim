import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from Solver import HALF_STENCIL, FiniteDifferenceTransport, FiniteVolume

GHOST_INDEX = {'in': 0, 'ext': -1}
FIRST_PHYS_VAL_INDEX = {'in': 1, 'ext': -2}

class NodeBase:

    """Docstring for Node. """

    def __init__(self):
        """TODO: to be defined. """
        self.neighbours = None
        self.neighbour_faces = None

    def set_neighbours(self, neighbours, faces):
        self.neighbours = neighbours
        self.neighbour_faces = faces
        assert self.neighbours.keys() == self.neighbour_faces.keys()

    def get_boundary_value(self, loc):
            raise NotImplementedError

    def update(self):
        raise NotImplementedError


class Node1D(NodeBase):

    """Docstring for Node1D. """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()
        # assert len(self.neighbours) == 2

    def get_boundary_gradient(self, loc):
        raise NotImplementedError


class Material:

    """Docstring for Air. """

    def __init__(self, cp, density, thermal_conductivty):
        """TODO: to be defined. """

        self.cp = cp
        self.density = density
        self.thermal_conductivty = thermal_conductivty
        self.diffusivity = self.compute_diffusivity()

    def compute_diffusivity(self):
        return self.thermal_conductivty / (self.cp * self.density)


class ConstantComponent(NodeBase):

    """Docstring for Air. """

    def __init__(self, y0):
        """TODO: to be defined. """

        super().__init__()
        self.y = y0

    def get_boundary_value(self, loc):
        return self.y

    def get_boundary_gradient(self, loc):
        return 0.

    def update(self):
        pass


# TODO define an equation as attribute
class Component(Node1D):

    """Docstring for Component2D. 
    number of ghost node on each side is HALF_STENCIL.
    There are (resolution) physical nodes, plus (2 * HALF_STENCIL) ghost nodes.
    There are (resolution-1) cells in the physical range.
    Normal orientation is: in--->ext
    """

    def __init__(
        self,
        name,
        material,
        thickness,
        y0,
        physics,
        boundary_type={'in': 'dirichlet', 'ext': 'dirichlet'},
        resolution=10,
        surface=1.0,
        observer=None,
    ):
        """TODO: to be defined. """

        super().__init__()
        self.name = name
        self.material = material
        assert(resolution > 1)
        self.resolution = resolution
        self.thickness = thickness
        self.surface = surface
        self.volume = thickness * surface
        self.y = np.zeros((self.resolution + 2 * HALF_STENCIL))
        assert len(y0) == resolution
        self.y[HALF_STENCIL:resolution+HALF_STENCIL] = y0[:]
        self.physics = physics
        self.sources = np.zeros((self.resolution))
        self.dx = self.thickness / (self.resolution - 1)
        self.boundary_loc = {'in': 0, 'ext': resolution-1}
        self.boundary_type = boundary_type
        self.observer = observer
        if self.observer is not None:
            observer.set_output_container(self)

    def get_physics_y(self):
        return self.y[HALF_STENCIL:self.resolution+HALF_STENCIL]

    def setGhostValue(self, face, val):
        self.y[GHOST_INDEX[face]] = val

    # TODO rename get_flux, move to FiniteDifferenceTransport, pass the diffusivity.
    # FiniteDifferenceTransport computes the derivative.
    def get_boundary_gradient(self, loc):
        return (self.y[GHOST_INDEX[loc]] - self.get_boundary_value(loc)) / self.dx

    def get_boundary_value(self, loc):
        return self.y[FIRST_PHYS_VAL_INDEX[loc]]

    # TODO rename update_ghost_node
    def update(self):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)

        for face, neigh in self.neighbours.items():
           if self.boundary_type[face] == 'dirichlet':
               # fill ghost node with the neighbour first physical node value.
               # self.setGhostValue(face, neigh.y[FIRST_PHYS_VAL_INDEX[self.neighbour_faces[face]]])
               self.setGhostValue(face, neigh.get_boundary_value(self.neighbour_faces[face]))
           elif self.boundary_type['in'] == 'adiabatic':
               # fill ghost node with first physical node value.
               # TODO implement a get_first_physical_value function
               self.setGhostValue(face, self.y[FIRST_PHYS_VAL_INDEX[face]])
           else:
               raise ValueError

    def update_sources(self, time):
        pass

