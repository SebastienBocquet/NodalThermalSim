import copy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from anytree import Node, RenderTree
from Solver import HALF_STENCIL
from Physics import FiniteDifferenceTransport, FiniteVolume


X = 0
Y = 1
Z = 2


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


class BoundaryCondition:

    def __init__(self, type, flux=None):
        self.type = type
        self.flux = flux

    def compute(self, neigh, neighbour_face, boundary_value, dx, thermal_conductivity):
        if self.type == 'dirichlet':
            # fill ghost node with the neighbour first physical node value, corrected to impose
            # a gradient that ensures heat flux conservation through component interface.
            ghost_val = neigh.get_boundary_value(neighbour_face)
            if neigh.material is not None:
                gradient_neighbour = (neigh.get_boundary_value(neighbour_face) -
                                      neigh.get_first_phys_value(neighbour_face)) / dx
                flux_neighbour = neigh.material.thermal_conductivty * gradient_neighbour
                flux_target = -flux_neighbour
                gradient = flux_target / thermal_conductivity
                ghost_target = boundary_value - dx * gradient
                error = ghost_val - ghost_target
                ghost_val += 1. * (ghost_val - ghost_target)
                # real_flux = -self.material.thermal_conductivity * (ghost_val - self.get_boundary_value(face)) / self.dx
                # if ite % 10000 == 0:
                #     print('error on ghost temperature', error)
                #     print('flux', real_flux)
                #     print('flux neighbour', flux_neighbour)
            return ghost_val
        elif self.type == 'adiabatic':
            # fill ghost node with first physical node value.
            return boundary_value
        elif self.type == 'flux':
            gradient = self.flux / thermal_conductivity
            ghost_val = boundary_value - dx * gradient
            return ghost_val
        else:
            raise ValueError


class NodeBase(ABC):

    GHOST_INDEX = {'left': 0, 'right': -1}
    BOUNDARY_VAL_INDEX = {'left': 1, 'right': -2}
    FIRST_PHYS_VAL_INDEX = {'left': 2, 'right': -3}

    def __init__(self, name, resolution, y0, delta_x, dx=-1):
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
        self.name = name
        self.material = None
        if dx > 0:
            self.dx = dx
            self.resolution = (int)(delta_x / dx + 1)
        else:
            self.resolution = resolution
            self.dx = delta_x / (self.resolution - 1)
        assert(self.resolution > 1)
        self.neighbours = {'x': None, 'y': None, 'z': None}
        self.neighbours_faces = {'x': None, 'y': None, 'z': None}
        self.neighbours_faces = {'x': None, 'y': None, 'z': None}
        self.boundary = None

    def check(self):
        assert self.boundary.keys() == self.GHOST_INDEX.keys()
        assert self.BOUNDARY_VAL_INDEX.keys() == self.GHOST_INDEX.keys()

    def check_stability(self, dt):
        if (dt / self.dx ** 2) >= 1.0 / (2 * self.material.diffusivity):
            raise ValueError

    def add_to_tree(self, node):
        node_ = Node(self.name, parent=node)
        for n in self.neighbours.values():
            if n is not None:
                Node(n.name, parent=node_)

    def get_physics_x(self):
        return self.x[HALF_STENCIL:self.resolution+HALF_STENCIL]

    def get_physics_y(self):
        return self.y[HALF_STENCIL:self.resolution+HALF_STENCIL]

    def set_neighbours(self, neighbours, faces={'left': 'right', 'right': 'left'}):
        self.neighbours = neighbours
        self.neighbour_faces = faces
        assert self.neighbours.keys() == self.neighbour_faces.keys()

    @abstractmethod
    def get_boundary_value(self, loc):
        return

    @abstractmethod
    def get_first_phys_value(self, loc):
        return

    @abstractmethod
    def get_boundary_gradient(self, loc):
        return

    def get_ghost_value(self, loxis='x'):
        return self.val[axis][self.GHOST_INDEX[loc]]

    def get_boundary_value(self, loc, axis='x'):
        return self.val[axis][self.BOUNDARY_VAL_INDEX[loc]]

    def get_first_phys_value(self, loc, axis='x'):
        return self.val[axis][self.FIRST_PHYS_VAL_INDEX[loc]]

    def update_ghost_node(self, time, ite, axis='x'):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)
        self.source.update(time)
        for face, neigh in self.neighbours[axis].items():
            ghost_val = self.boundary[face].compute(neigh, self.neighbour_faces[axis][face],
                                                    self.get_boundary_value(face, axis),
                                                    self.dx, self.material.thermal_conductivty)
            self.setGhostValue(face, ghost_val, axis)


class ConstantComponent(NodeBase):

    """Docstring for Air. """

    RESOLUTION = 2
    DELTA_X = 1.

    def __init__(self, name, y0):
        """TODO: to be defined. """

        super().__init__(name, self.RESOLUTION, y0, self.DELTA_X)
        self.name = name
        self.y = y0
        self.material = None

    def get_boundary_value(self, loc):
        return self.y

    def get_boundary_gradient(self, loc):
        return 0.

    def get_first_phys_value(self, loc):
        return self.y

    def update_ghost_node(self, time, ite):
        pass


class Component(NodeBase):

    """
    Component1D
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
        outputs,
        boundary={'left': BoundaryCondition('dirichlet'), 'right': BoundaryCondition('dirichlet')},
        resolution=10,
        dx=-1,
        surface=1.0,
        source=None,
    ):

        super().__init__(name, resolution, y0, thickness, dx)
        self.material = material
        self.thickness = thickness
        self.surface = surface
        self.volume = thickness * surface
        self.physics = physics
        if source is None:
            self.source = Source(0., self.resolution)
        else:
            self.source = source
        self.boundary = boundary
        self.outputs = outputs
        x0 = 0.
        self.x = {'x': np.linspace(x0 - HALF_STENCIL * self.dx,
                             (self.resolution - 1 + HALF_STENCIL) * self.dx,
                             num=self.resolution + 2 * HALF_STENCIL)}
        self.val = {'x': np.zeros((self.resolution + 2 * HALF_STENCIL))}
        self.val['x'][HALF_STENCIL:self.resolution + HALF_STENCIL] = y0
        logger.info(f'Component name is {self.name}')
        logger.info(f'Component discretization step (m) is {self.dx}')
        logger.info(f'Component diffusivity is {self.material.diffusivity}')

    def check_stability(self, dt):
        if (dt / self.dx ** 2) >= 1.0 / (2 * self.material.diffusivity):
            raise ValueError

    def check(self):
        assert self.boundary.keys() == self.ghost_index.keys()
        assert self.boundary_val_index.keys() == self.ghost_index.keys()

    def setGhostValue(self, face, val):
        self.y[self.ghost_index[face]] = val

    def setBoundaryValue(self, face, val):
        self.y[self.boundary_val_index[face]] = val

    # gradient is oriented towards the exterior of the component.
    def get_boundary_gradient(self, loc):
        return (self.y[self.ghost_index[loc]] - self.get_boundary_value(loc)) / self.dx

    def get_ghost_value(self, loc):
        return self.y[self.ghost_index[loc]]

    def get_boundary_value(self, loc):
        return self.y[self.boundary_val_index[loc]]

    def get_first_phys_value(self, loc):
        return self.y[self.first_phys_val_index[loc]]

    # TODO rename update_ghost_node
    def update_ghost_node(self, time, ite):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)
        self.source.update(time)
        for face, neigh in self.neighbours.items():
            ghost_val = self.boundary[face].compute(neigh, self.neighbour_faces[face],
                                                    self.get_boundary_value(face),
                                                    self.dx, self.material.thermal_conductivty)
            self.setGhostValue(face, ghost_val)


class Box(NodeBase):

    """Docstring for Room. 
    """

    RESOLUTION = 1

    def __init__(
        self,
        name,
        material,
        delta_x,
        delta_y,
        delta_z,
        y0,
        physics,
        outputs,
        boundary={
            'x': {'left': BoundaryCondition('dirichlet'),
                  'right': BoundaryCondition('dirichlet')},
            'y': {'left': BoundaryCondition('dirichlet'),
                  'right': BoundaryCondition('dirichlet')},
            'z': {'left': BoundaryCondition('dirichlet'),
                  'right': BoundaryCondition('dirichlet')},
        },
        source=None
    ):

        # resolution is computed based on x axis
        super().__init__(name, self.RESOLUTION, y0, delta_x, dx-1)
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        self.material = material
        self.volume = delta_x * delta_y * delta_z
        self.physics = physics
        if source is None:
            self.source = Source(0., self.RESOLUTION)
        else:
            self.source = source
        self.boundary = boundary
        self.ghost_index = GHOST_INDEX
        self.boundary_val_index = BOUNDARY_VAL_INDEX
        self.first_phys_val_index = FIRST_PHYS_VAL_INDEX
        self.outputs = outputs


def create_component(
    type,
    name,
    material,
    dimensions,
    y0,
    outputs,
    boundary_type={'left': 'dirichlet', 'right': 'dirichlet'},
    resolution=10,
    flux={'left': None, 'right': None},
):

    if type == '1D':
        physics_default = FiniteDifferenceTransport()
        c = Component(name, material, dimensions[X], y0, physics_default(),
                      outputs, boundary_type, resolution,
                      surface=dimensions[Y] * dimensions[Z],
                      source=None, flux=flux)
        return c
    else:
        raise ValueError()


