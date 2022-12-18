import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HALF_STENCIL = 1
T0 = 273.15


class OutputComputerBase():

    def __init__(self):
        pass

    def compute_var(self):
        return NotImplementedError


def OUTPUT_SIZE(var_name, resolution):
    """
    Returns the size of the output.

    The user must define here the output size for every variable defined
    in class OutputComputer.

    Parameters:
    var_name (str): name of the output variable.
    resolution (int): resolution (length of the discretized axis)
    of the component from which the variable is post-processed.

    Returns:
    int: the size of the variable, ie the length of the axis
    on which the variable values are stored.

    """

    if var_name == 'temperature':
        return resolution
    elif var_name == 'temperature_gradient':
        return resolution - 1
    elif var_name == 'heat_flux':
        return resolution - 1
    elif var_name == 'HTC':
        return 1


class OutputComputer(OutputComputerBase):

    """Handle the size of the output data,
    and according to the location,
    computes the output based on Component raw data. """

    def __init__(self):
        pass


    def compute_var(self, c, output):
        if output.var_name == 'temperature':
            if output.loc == 'all':
                return c.get_grid().get_physics_val()
            else:
                return np.array([c.get_grid().get_boundary_value(output.loc)])
        elif output.var_name == 'temperature_gradient':
            if output.loc == 'all':
                return np.diff(c.get_grid().get_physics_val()) / c.get_grid().dx
            else:
                return np.array([c.get_grid().get_boundary_gradient(output.loc)])
        elif output.var_name == 'heat_flux':
            if output.loc == 'all':
                return c.material.thermal_conductivity * np.diff(c.get_grid().get_physics_val() / c.get_grid().dx)
            else:
                return np.array([c.material.thermal_conductivity * c.get_grid().get_boundary_gradient(output.loc)])
        # elif output.var_name == 'HTC':
        #     surface_temperature = 0.
        #     ref_temperature = 0.
        #     if output.loc == 'in':
        #         # TODO introduce a get_ghost_value()[output.loc]
        #         surface_temperature = 0.5 * (c.y[0] + c.get_grid().get_physics_val()[0])
        #         ref_temperature = c.get_grid().get_physics_val()[0]
        #     else:
        #         # TODO introduce a get_ghost_value()[output.loc]
        #         surface_temperature = 0.5 * (c.get_grid().get_physics_val()[-1] + c.y[c.resolution + 1])
        #         ref_temperature = c.get_grid().get_physics_val()[-1]
        #     htc = c.material.thermal_conductivity * c.get_boundary_gradient(output.loc) / (surface_temperature - ref_temperature)
        #     return np.array([htc])


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
                    error = ghost_target - ghost_val
                    ghost_val -= 1. * error
                return ghost_val, ghost_target
            elif self.type == 'non_conservative':
                return ghost_val,
            else:
                raise ValueError


class FiniteVolume:

    """Docstring for FiniteVolume.
    """

    def __init__(self):
        """TODO: to be defined. """

    def advance_time(self, dt, c, ite, solver_type):
        # update central value
        sum_heat_flux = 0.
        for ax, grid in c.grid.items():
            for face, neigh in grid.neighbours.items():
                heat_flux = grid.get_boundary_heat_flux(face, ax)
                sum_heat_flux += heat_flux * c.get_surface(ax)
        temp_variation = sum_heat_flux / (c.material.density * c.material.cp * c.volume)
        for ax, grid in c.grid.items():
            grid.val[grid.FIRST_PHYS_VAL_INDEX['left']] += temp_variation * dt

        # update boundary values
        for ax, grid in c.grid.items():
            for face, neigh in grid.neighbours.items():
                grid.val[grid.BOUNDARY_VAL_INDEX[face]] = \
                    0.5 * (grid.get_ghost_value(face) +
                           grid.val[grid.FIRST_PHYS_VAL_INDEX[face]])


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    def __init__(self):
        """TODO: to be defined. """

        self.Ainv = None

    def advance_time(self, dt, c, ite, solver_type):
        # TODO set dt and solver_type at beginning of solver
        if solver_type == 'implicit':
            r = dt * c.material.diffusivity / c.get_grid().dx**2
            if ite == 0:
                N = c.get_grid().resolution
                A = np.zeros((N, N))
                for i in range(0, N):
                    A[i, i] = 1 + 2 * r
                    # for flux imposed BC
                    if type(c.get_grid().boundary['left']) == BoundaryConditionFlux:
                        A[0, 0] = 1 + r
                    if type(c.get_grid().boundary['right']) == BoundaryConditionFlux:
                        A[-1, -1] = 1 + r

                for i in range(0, N - 1):
                    A[i + 1, i] = -r
                    A[i, i + 1] = -r

                self.Ainv = np.linalg.inv(A)
                # logger.log(logging.INFO, f"A: {A}")
                # print('A', A)
                # print('inv A', self.Ainv)
            val_ = c.get_grid().val
            w = val_[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL]
            # for flux imposed BC
            if type(c.get_grid().boundary['left']) == BoundaryConditionFlux:
                k_left = -c.get_grid().dx * c.get_grid().boundary['left'].flux / c.material.thermal_conductivity
                w[0] += r * k_left
            else:
                w[0] += r * val_[0]
            if type(c.get_grid().boundary['right']) == BoundaryConditionFlux:
                k_right = -c.get_grid().dx * c.get_grid().boundary['right'].flux / c.material.thermal_conductivity
                w[-1] += r * k_right
            else:
                w[-1] += r * val_[-1]
            # print('rhs', w)
            val_[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL] = np.dot(self.Ainv, w)
            # print('updated vals', val_[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL])
        elif solver_type == 'explicit':
            dydx = np.diff(c.get_grid().val)
            diffusion = c.material.diffusivity * (dydx[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL] -
                                                           dydx[:-HALF_STENCIL]) / c.get_grid().dx ** 2
            c.get_grid().val[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL] += dt * (diffusion[:] + c.source.y[:])
        else:
            raise TypeError

