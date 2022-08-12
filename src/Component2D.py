import matplotlib.pyplot as plt
import numpy as np


class NodeBase:

    """Docstring for Node. """

    def __init__(self, neighbours):
        """TODO: to be defined. """
        self.neighbours = neighbours

    def get_neighbour_val(self, loc):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class Node1D(NodeBase):

    """Docstring for Node1D. """

    def __init__(self, neighbours):
        """TODO: to be defined. """
        super().__init__(neighbours)
        assert len(self.neighbours) == 2

    neighbour_loc = ['in', 'ext']


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


class ConstantComponent:

    """Docstring for Air. """

    def __init__(self, y0):
        """TODO: to be defined. """

        self.y = y0

    def get_neighbour_val(self, loc):
        return self.y


class Component2D(Node1D):

    """Docstring for Component2D. 
    in           ext
    y[0]-------->y[resolution-1]
    """

    def __init__(
        self,
        material,
        thickness,
        y0,
        component_in_neighbour,
        component_ext_neighbour,
        neighbours,
        observer=None,
        resolution=10,
        surface=1.0
    ):
        """TODO: to be defined. """

        super().__init__(neighbours)
        self.material = material
        # TODO introduce a STENCIL constant.
        self.resolution = resolution
        # self.component_in_neighbour = component_in_neighbour
        # self.component_ext_neighbour = component_ext_neighbour
        self.thickness = thickness
        self.surface = surface
        self.y = np.zeros((self.resolution + 2)) + y0
        self.sources = np.zeros((self.resolution))
        self.dx = self.thickness / self.resolution
        self.observer = observer
        if observer is not None:
            assert self.observer.result.shape[0] == self.resolution + 2

    def compute_wall_heat_flux(self, y):
        assert len(y) == 2
        return -self.material.thermal_conductivty * (y[1] - y[0]) / self.dx

    # TODO rename get_flux, move to FiniteDifferenceTransport, pass the diffusivity.
    # FiniteDifferenceTransport computes the derivative.
    def get_wall_heat_flux_in(self,):
        y_border = self.y[:2]
        wall_heat_flux_in = self.compute_wall_heat_flux(y_border)
        return wall_heat_flux_in

    def get_wall_heat_flux_ext(self,):
        y_reverse = self.y[::-1]
        y_border = y_reverse[:2]
        wall_heat_flux_ext = self.compute_wall_heat_flux(y_border)
        return wall_heat_flux_ext

    def get_neighbour_val(self, loc):
        if loc == 'in':
            return self.y[1]
        elif loc == 'ext':
            return self.y[self.resolution]
        else:
            raise ValueError

    def update(self):
        self.y[0] = self.neighbours['in'].get_neighbour_val('ext')
        self.y[self.resolution + 1] = self.neighbours['ext'].get_neighbour_val('in')

    def update_sources(self, time):
        pass

