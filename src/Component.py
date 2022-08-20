import matplotlib.pyplot as plt
import numpy as np
from Solver import HALF_STENCIL, FiniteDifferenceTransport, FiniteVolume


class NodeBase:

    """Docstring for Node. """

    def __init__(self):
        """TODO: to be defined. """
        self.neighbours = None

    def set_neighbours(self, n):
        self.neighbours = n

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


# TODO define an equation as attribute
class Box(NodeBase):

    """Docstring for Box. """

    def __init__(
        self,
        name,
        material,
        y0,
        volume=1.0,
        observer=None
    ):
        """TODO: to be defined. """

        # assert len(self.neighbours) == 6
        self.name = name
        self.material = material
        self.volume = volume
        self.dx = volume ** (1. / 3)
        self.y = np.ones((1)) * y0
        self.physics = FiniteVolume()
        self.sources = 0.
        self.gradients = np.zeros((6))
        self.observer = observer
        if self.observer is not None:
            observer.set_resolution(1)
        #TODO set dt of observer here.
        # put observer in NodeBase.
        if observer is not None:
            assert self.observer.result.shape[0] == 1

    # TODO should have a get_boundary_value such that linked components can be updated

    def update(self):
        pass

    #TODO store gradients (or flux) in a list.
    # Add boundary_type. Handle case of adiabatic boundary (set to zero flux).
    def get_boundary_value(self, loc):
        return self.y[0]

    def get_boundary_gradient(self, loc):
        return self.gradients[loc]

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
        self.y = np.zeros((self.resolution + 2 * HALF_STENCIL))
        self.y[HALF_STENCIL:resolution+HALF_STENCIL] = y0[:]
        self.physics = FiniteDifferenceTransport()
        self.sources = np.zeros((self.resolution))
        self.dx = self.thickness / (self.resolution - 1)
        self.boundary_type = boundary_type
        self.observer = observer
        if self.observer is not None:
            observer.set_resolution(resolution + 2 * HALF_STENCIL)
        #TODO set dt of observer here.
        if observer is not None:
            assert self.observer.result.shape[0] == self.resolution + 2 * HALF_STENCIL

    def __compute_gradient(self, y):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        assert len(y) == 2
        return -(y[1] - y[0]) / self.dx

    # TODO rename get_flux, move to FiniteDifferenceTransport, pass the diffusivity.
    # FiniteDifferenceTransport computes the derivative.
    def get_boundary_gradient(self, loc):
        if loc == 'in':
            y_border = self.y[:2]
            wall_heat_flux_in = self.__compute_gradient(y_border)
            return wall_heat_flux_in
        elif loc == 'ext':
            y_reverse = self.y[::-1]
            y_border = y_reverse[:2]
            wall_heat_flux_ext = self.__compute_gradient(y_border)
            return wall_heat_flux_ext
        else:
            raise ValueError

    def get_boundary_value(self, loc):
        if loc == 'in':
            return self.y[1]
        elif loc == 'ext':
            return self.y[self.resolution]
        else:
            raise ValueError

    # TODO rename update_ghost_node
    def update(self):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)

        # TODO: loop on the two boundaries
        if self.boundary_type['in'] == 'dirichlet':
            # fill ghost node with the neighbour first physical node value.
            self.y[0] = self.neighbours['in'].get_boundary_value('ext')
        elif self.boundary_type['in'] == 'adiabatic':
            # fill ghost node with first physical node value.
            # TODO implement a get_first_physical_value function
            self.y[0] = self.y[1]
        else:
            raise ValueError

        if self.boundary_type['ext'] == 'dirichlet':
            self.y[self.resolution + 1] = self.neighbours['ext'].get_boundary_value('in')
        elif self.boundary_type['ext'] == 'adiabatic':
            self.y[self.resolution + 1] = self.y[self.resolution]
        else:
            raise ValueError

    def update_sources(self, time):
        pass

