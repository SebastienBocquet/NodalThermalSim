import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from anytree import Node, RenderTree
from Solver import HALF_STENCIL
from Physics import FiniteDifferenceTransport, FiniteVolume

GHOST_INDEX = {'left': 0, 'right': -1}
FIRST_PHYS_VAL_INDEX = {'left': 1, 'right': -2}


class Source():

    """Docstring for Source. 
    """

    def __init__(
        self,
        y0,
        resolution
    ):

        self.y0 = y0
        self.resolution = resolution
        self.y = self.y0 * np.ones((resolution))

    def update(self, time):
        pass


class NodeBase:

    """Docstring for Node. """

    def __init__(self):
        """TODO: to be defined. """
        self.neighbours = None
        self.neighbour_faces = None

    def set_neighbours(self, neighbours, faces={'left': 'right', 'right': 'left'}, parent_node=None):
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

    def __init__(self, name, y0):
        """TODO: to be defined. """

        super().__init__()
        self.name = name
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
    Normal orientation is: left--->right
    """

    def __init__(
        self,
        name,
        material,
        thickness,
        y0,
        physics,
        boundary_type={'left': 'dirichlet', 'right': 'dirichlet'},
        resolution=10,
        dx=-1,
        surface=1.0,
        source=None,
        observer=None,
    ):
        """TODO: to be defined. """

        super().__init__()
        self.name = name
        self.material = material
        self.thickness = thickness
        if dx > 0:
            self.dx = dx
            self.resolution = (int)(thickness / dx + 1)
        else:
            self.resolution = resolution
            self.dx = self.thickness / (self.resolution - 1)
        assert(self.resolution > 1)
        self.surface = surface
        self.volume = thickness * surface
        self.y = np.zeros((self.resolution + 2 * HALF_STENCIL))
        # assert len(y0) == self.resolution
        self.y[HALF_STENCIL:self.resolution+HALF_STENCIL] = y0
        self.physics = physics
        physics.y_mean_conservative = y0
        if source is None:
            self.source = Source(0., self.resolution)
        else:
            self.source = source
        self.boundary_type = boundary_type
        self.ghost_index = GHOST_INDEX
        self.first_phys_val_index = FIRST_PHYS_VAL_INDEX
        self.observer = observer
        if self.observer is not None:
            observer.set_output_container(self)
        self.node = Node(name)

    def show_tree(self):
        for n in self.neighbours.values():
            Node(n.name, parent=self.node)
        print("\n")
        for pre, fill, node in RenderTree(self.node):
            print("%s%s" % (pre, node.name))

    def check(self):
        # __import__('pudb').set_trace()
        assert self.boundary_type.keys() == self.ghost_index.keys()
        assert self.first_phys_val_index.keys() == self.ghost_index.keys()

    def get_physics_y(self):
        return self.y[HALF_STENCIL:self.resolution+HALF_STENCIL]

    def setGhostValue(self, face, val):
        self.y[self.ghost_index[face]] = val

    # TODO rename get_flux, move to FiniteDifferenceTransport, pass the diffusivity.
    # FiniteDifferenceTransport computes the derivative.
    def get_boundary_gradient(self, loc):
        return (self.y[self.ghost_index[loc]] - self.get_boundary_value(loc)) / self.dx

    def get_boundary_value(self, loc):
        return self.y[self.first_phys_val_index[loc]]

    # TODO rename update_ghost_node
    def update(self, time):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)
        self.source.update(time)
        for face, neigh in self.neighbours.items():
           if self.boundary_type[face] == 'dirichlet':
               # fill ghost node with the neighbour first physical node value.
               self.setGhostValue(face, neigh.get_boundary_value(self.neighbour_faces[face]))
           elif self.boundary_type[face] == 'adiabatic':
               # fill ghost node with first physical node value.
               self.setGhostValue(face, self.y[self.first_phys_val_index[face]])
           else:
               raise ValueError


class Room(Component):

    """Docstring for Room. 
    """

    def __init__(
        self,
        name,
        material,
        thickness,
        y0,
        physics,
        boundary_type={'left': 'dirichlet', 'right': 'dirichlet'},
        resolution=10,
        dx=-1,
        surface=1.0,
        source=None,
        observer=None,
    ):
        """TODO: to be defined. """

        super().__init__(
            name,
            material,
            thickness,
            y0,
            physics,
            boundary_type,
            resolution,
            dx,
            surface,
            source,
            observer,
        )



