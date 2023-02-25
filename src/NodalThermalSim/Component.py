import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
import numpy as np
from anytree import Node, RenderTree
from NodalThermalSim.Solver import HALF_STENCIL
from NodalThermalSim.Physics import FiniteDifferenceTransport, FiniteVolume
from NodalThermalSim.Grid import GridBase, BoundaryConditionFlux


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

    """Docstring for Material. """

    def __init__(self, cp, density, thermal_conductivty):
        """TODO: to be defined. """

        self.cp = cp
        self.density = density
        self.thermal_conductivity = thermal_conductivty
        self.diffusivity = self.compute_diffusivity()

    def compute_diffusivity(self):
        return self.thermal_conductivity / (self.cp * self.density)


class NodeBase(ABC):

    def __init__(self, name, surface):
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
        self.surface = surface

    def get_name(self):
        return self.name

    def get_surface(self, ax='x'):
        return self.surface

    @abstractmethod
    def check_stability(self, dt):
        ...

    @abstractmethod
    def check(self):
        ...

    def add_to_tree(self, node):
        node_ = Node(self.name, parent=node)
        for n in self.get_grid().neighbours.values():
            if n is not None:
                Node(n.name, parent=node_)

    @abstractmethod
    def get_grid(self, ax='x') -> GridBase:
        ...


class ConstantComponent(NodeBase):

    """Docstring for Air. """

    RESOLUTION = 2
    DELTA_X = 1.

    def __init__(self, name, y0, surface=1.):
        """TODO: to be defined. """

        super().__init__(name, surface)
        self.name = name
        self.y = y0
        self.material = None
        self.grid = GridBase(self.RESOLUTION, y0, self.DELTA_X)

    def check_stability(self, dt):
        pass

    def check(self):
        pass

    def get_grid(self, ax='x'):
        return self.grid

    def update_ghost_node(self, time, ite):
        pass


class Component1D(NodeBase):

    """
    Component1D
    number of ghost node on each side is HALF_STENCIL.
    There are (resolution) physical nodes, plus (2 * HALF_STENCIL) ghost nodes.
    There are (resolution-1) cells in the physical range.
    Normal orientation is: left--->right
    """

    SOURCE_BOUNDARY_VALUE = {'left': 0, 'right': -1}

    def __init__(
        self,
        name,
        material,
        thickness,
        y0,
        physics,
        outputs = None,
        resolution=10,
        dx=-1,
        surface=1.0,
        source=None,
    ):

        super().__init__(name, surface)
        self.grid = GridBase(resolution, y0, thickness, dx)
        self.material = material
        self.surface = surface
        self.volume = thickness * surface
        self.physics = physics
        if source is None:
            self.source = Source(0., self.get_grid().resolution)
        else:
            self.source = source
        self.outputs = outputs

    def check_stability(self, dt):
        if (dt / self.get_grid().dx ** 2) >= 1.0 / (2 * self.material.diffusivity):
            raise ValueError

    def set_outputs(self, outputs):
        self.outputs = outputs

    def check(self):
        pass

    def get_grid(self, ax='x') -> GridBase:
        return self.grid

    def update_ghost_node(self, time, ite, ax='x'):
        # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
        # use a list of opposite index to automatically access the other link.
        assert(HALF_STENCIL == 1)
        self.source.update(time)
        for face, neigh in self.get_grid(ax).neighbours.items():
            neigh_face = self.get_grid(ax).neighbour_faces[face]
            ghost_val, conservative_corr = self.get_grid(ax).boundary[face].compute(ite, face, neigh, neigh_face,
                                                                  self.get_grid(ax).get_boundary_value(face),
                                                                  self.get_grid(ax).get_first_phys_value(face),
                                                                  self.get_grid(ax).dx, self.material.thermal_conductivity,
                                                                  self.name)
            self.get_grid(ax).setGhostValue(face, ghost_val)
            self.source.y[self.SOURCE_BOUNDARY_VALUE[face]] = 0.5 * conservative_corr
            if neigh is not None and neigh.material is not None:
                neigh.source.y[neigh.SOURCE_BOUNDARY_VALUE[neigh_face]] = -0.5 * conservative_corr


class Box(NodeBase):

    """Docstring for Room. 
    """

    RESOLUTION = 3

    SOURCE_BOUNDARY_VALUE = {'left': 0, 'right': -1}

    def __init__(
        self,
        name,
        material,
        delta_x,
        delta_y,
        delta_z,
        y0,
        outputs,
        source=None
    ):

        # TODO: create base object which has common interface with Box and Component1D
        # TODO create a 1D grid object which handles the discretization and ghost nodes.
        super().__init__(name, surface=0.)
        self.material = material
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        self.volume = delta_x * delta_y * delta_z
        self.physics = FiniteVolume()
        if source is None:
            self.source = Source(0., self.RESOLUTION)
        else:
            self.source = source
        self.outputs = outputs
        x0 = 0.
        dx = delta_x / (self.RESOLUTION - 1)
        dy = delta_y / (self.RESOLUTION - 1)
        dz = delta_z / (self.RESOLUTION - 1)
        self.axis = ['x', 'y', 'z']
        self.grid = {
            'x': GridBase(self.RESOLUTION, y0, delta_x,  dx),
            'y': GridBase(self.RESOLUTION, y0, delta_y,  dy),
            'z': GridBase(self.RESOLUTION, y0, delta_z,  dz),
        }
        for g in self.grid.values():
            g.set_boundary({'left': BoundaryConditionFlux(),
                            'right': BoundaryConditionFlux()})
        self.surface = {
            'x': delta_y * delta_z,
            'y': delta_x * delta_z,
            'z': delta_x * delta_y,
        }

    def check_stability(self, dt):
        pass

    def check(self):
        """
        Check that if:
          - the boundary condition of the box face is a heat transfer coefficient boundary
          - the neighbour attached to this face is a Component1D
        then the boundary condition of the neighbour on the corresponding face is also a
        transfer coefficient boundary.

        """
        # for ax, grid in self.grid.items():
        #     for face in grid.BOUNDARY_VAL_INDEX.keys():
        #         bnd = grid.boundary[face]
        #         assert type(bnd) is BoundaryConditionFlux
        pass

        # for ax, grid in self.grid.items():
        #     for face in grid.BOUNDARY_VAL_INDEX.keys():
        #         bnd = grid.boundary[face]
        #         if (type(bnd) is BoundaryConditionFlux) and bnd.type == 'htc':
        #             neighbour = grid.neighbours[face]
        #             if type(neighbour) is Component1D:
        #                 grid_neighbour = neighbour.get_grid(ax)
        #                 bnd_neighbour = grid_neighbour.boundary[grid_neighbour.neighbour_faces[face]]
        #                 assert (type(bnd_neighbour) is BoundaryConditionFlux) and bnd_neighbour.type == 'htc'

    def get_grid(self, ax='x') -> GridBase:
        return self.grid[ax]

    def get_surface(self, ax):
        return self.surface[ax]

    def update_ghost_node(self, time, ite):
        assert(HALF_STENCIL == 1)
        self.source.update(time)
        for ax, grid in self.grid.items():
            # this implementation works only for 1 ghost point, ie HALF_STENCIL=1.
            # use a list of opposite index to automatically access the other link.
            for face, neigh in grid.neighbours.items():
                ghost_val, boundary_val = grid.boundary[face].compute_box(ite, face, neigh, grid.neighbour_faces[face],
                                                         grid.get_boundary_value(face),
                                                         grid.get_first_phys_value(face),
                                                         grid.dx, self.material.thermal_conductivity,
                                                         self.name)
                grid.setGhostValue(face, ghost_val)
                if boundary_val is not None:
                    grid.setBoundaryValue(face, boundary_val)


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
        c = Component1D(name, material, dimensions[X], y0, physics_default(),
                        outputs, boundary_type, resolution,
                        surface=dimensions[Y] * dimensions[Z],
                        source=None, flux=flux)
        return c
    else:
        raise ValueError()
