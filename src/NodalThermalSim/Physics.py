import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HALF_STENCIL = 1
T0 = 273.15

DISPLAY_PERIOD = 10000

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
                return c.get_grid().get_physics_val() - T0
            else:
                return np.array([c.get_grid().get_boundary_value(output.loc)]) - T0
        elif output.var_name == 'temperature_gradient':
            if output.loc == 'all':
                return np.diff(c.get_grid().get_physics_val()) / c.get_grid().dx
            else:
                return np.array([c.get_grid().get_boundary_gradient(output.loc)])
        elif output.var_name == 'heat_flux':
            if output.loc == 'all':
                return -c.material.thermal_conductivity * np.diff(c.get_grid().get_physics_val() / c.get_grid().dx)
            else:
                return np.array([c.material.thermal_conductivity * c.get_grid().get_boundary_gradient(output.loc)])


class BoundaryConditionFlux:

    NB_DX_DIST_BND_TO_REF_PT = 3

    def __init__(self, type='heat_flux', flux=0., htc=0., ref_temperature=None):
        """

        Args:
            type: type of boundary condition: heatFlux or htc (heat transfer coefficient,
            which means that the flux is equal to htc * the delta of temperature across the boundary.
            flux: value of imposed flux (W/m2) for boundary condition heat_flux.
            Flux is oriented inwards, so positive flux means a transfer of energy towards the component.
            htc: value of heat transfer coefficient for boundary condition htc.
            ref_temperature:
         """
        self.type = type
        self.flux = flux
        self.htc = htc
        self.ref_temperature = ref_temperature

    def compute(self, ite, face, neigh, neighbour_face, boundary_value, first_phys_val, dx, thermal_conductivity, name=None):
        # flux = - thermal_conductivity * gradient.
        # But gradient is oriented outwards, and flux is oriented inwards.
        # So flux = thermal_conductivity * gradient.
        if self.type == 'heat_flux':
            gradient = self.flux / thermal_conductivity
            # print('face', face)
            # print('gradient', gradient)
            ghost_val = boundary_value + dx * gradient
            # print('ghost val, boundary val', ghost_val, boundary_value)
            return ghost_val, 0.
        elif self.type == 'htc':
            gradient_neighbour = (neigh.get_grid().get_boundary_value(neighbour_face) -
                                  neigh.get_grid().get_first_phys_value(neighbour_face)) / neigh.get_grid().dx
            # neigh_value = neigh.get_grid().get_boundary_value(neighbour_face)
            neigh_value = neigh.get_grid().get_first_phys_value(neighbour_face)
            # from NodalThermalSim.Grid import GridBase
            # neigh_value = neigh.get_grid().get_physics_val()[GridBase.THIRD_PHYS_VAL_INDEX[face]]
            # neigh_value = np.mean(neigh.get_grid().get_physics_val())
            # gradient_neighbour = (boundary_value - ref_temperature) / dist_bnd_to_ref_pt
            if self.ref_temperature is not None:
                ref_temperature_ = self.ref_temperature
            else:
                ref_temperature_ = neigh_value
                # ref_temperature_ = boundary_value - gradient_neighbour * self.NB_DX_DIST_BND_TO_REF_PT * dx
            # delta = ref_temperature_ - np.mean(vals)
            delta = ref_temperature_ - first_phys_val
            self.flux = self.htc * delta
            gradient = self.flux / thermal_conductivity
            ghost_val = boundary_value + dx * gradient
            if ite % 1000 == 0:
                print(f'Flux BC for face {face} of component {name}: \n \
                      bnd temperature is {first_phys_val: .2f} \n \
                      ref temperature is {ref_temperature_: .2f} \n \
                      delta temperature is {delta: .2f}, \n \
                      flux is {self.flux: .2f}')
            return ghost_val, 0.
        else:
            raise ValueError

    def compute_box(self, ite, face, neigh, neighbour_face, boundary_value, vals, dx, thermal_conductivity, name=None):
        ghost_val = self.compute(ite, face, neigh, neighbour_face, boundary_value, vals, dx, thermal_conductivity, name)
        return ghost_val[0], None


class BoundaryConditionDirichlet:

    def __init__(self, type='conservative'):
        self.type = type

    def compute(self, ite, face, neigh, neighbour_face, boundary_value, first_phys_value, dx, thermal_conductivity, name=None):
            ghost_val = neigh.get_grid().get_boundary_value(neighbour_face)
            conservation_corr = 0.
            ghost_target = 0.
            if self.type == 'conservative':
                if neigh.material is None:
                    if ite % DISPLAY_PERIOD == 0:
                        logger.log(logging.WARNING, "Conservative Dirichlet boundary condition cannot be activated \
                                    because it requires that the neighbour component defines a material. \
                                    Switch to non conservative boundary.")
                    return ghost_val, 0.
                from NodalThermalSim.Component import Box
                if type(neigh) is Box:
                    # TODO introduce attribute axis for Component1D
                    raise ValueError('conservative dirichlet is not allowed adjacent to box component')
                else:
                    gradient_neighbour = (neigh.get_grid().get_boundary_value(neighbour_face) -
                                          neigh.get_grid().get_first_phys_value(neighbour_face)) / neigh.get_grid().dx
                    flux_neighbour = neigh.material.thermal_conductivity * gradient_neighbour
                    flux = thermal_conductivity * (boundary_value - first_phys_value) / dx
                conservation_corr = -(flux_neighbour - flux) / dx
                if ite % DISPLAY_PERIOD == 0:
                    msg = ''
                    print(f'Dirichlet BC for face {face} of component {name}: \n \
                          neighbour flux is {flux_neighbour: .2f} \n \
                          target ghost value is {ghost_target: .2f}, \n \
                          bnd value is {boundary_value: .2f}, \n \
                          conservation error is {conservation_corr: .2f}, \n \
                          ghost value is {ghost_val: .2f}, \n \
                          {msg}')
                return ghost_val, conservation_corr
            elif self.type == 'non_conservative':
                return ghost_val, 0.
            else:
                raise ValueError

    def compute_box(self, ite, face, neigh, neighbour_face, boundary_value, first_phys_value, dx, thermal_conductivity, name=None):
        # fill ghost node with the neighbour first physical node value.
        # Determine boundary value to satisfy the flux conservation through the boundary.
        ghost_val = neigh.get_grid().get_boundary_value(neighbour_face)
        gradient_neighbour = (neigh.get_grid().get_boundary_value(neighbour_face) -
                              neigh.get_grid().get_first_phys_value(neighbour_face)) / neigh.get_grid().dx
        flux_neighbour = neigh.material.thermal_conductivity * gradient_neighbour
        flux_target = -flux_neighbour
        gradient = flux_target / thermal_conductivity
        boundary_value_target = ghost_val - dx * gradient
        if ite % DISPLAY_PERIOD == 0:
            print(f'Dirichlet BC for face {face} of component {name}: \n \
                          neighbour flux is {flux_neighbour: .2f} \n \
                          target ghost value is {ghost_val: .2f}, \n \
                          bnd value is {boundary_value_target: .2f}')
        return ghost_val, boundary_value_target

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
                heat_flux = c.material.thermal_conductivity * grid.get_boundary_gradient(face) * c.get_surface(ax)
                # print(f'heat flux through face {face} axis {ax} is {heat_flux} with surface {c.get_surface(ax)}')
                sum_heat_flux += heat_flux
        # print('sum heat through box', sum_heat_flux)
        temp_variation = sum_heat_flux / (c.material.density * c.material.cp * c.volume)
        for ax, grid in c.grid.items():
            # first phys val left is equal to first phys val right, since there is only one phys value.
            grid.val[grid.FIRST_PHYS_VAL_INDEX['left']] += temp_variation * dt

        # update boundary values
        # for ax, grid in c.grid.items():
        #     for face, neigh in grid.neighbours.items():
        #         # grid.val[grid.BOUNDARY_VAL_INDEX[face]] = \
        #         #     0.5 * (grid.get_ghost_value(face) +
        #         #            grid.val[grid.FIRST_PHYS_VAL_INDEX[face]])
        #         grid.val[grid.BOUNDARY_VAL_INDEX[face]] = \
        #             grid.val[grid.FIRST_PHYS_VAL_INDEX['left']]


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
            w = val_[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL] + \
                c.source.y * dt / (c.material.density * c.material.cp)
            # for flux imposed BC, the ghost value at next time
            # k = w_ghost - w_bnd = dx * gradient = dx * flux / thermal_conductivity
            if type(c.get_grid().boundary['left']) == BoundaryConditionFlux:
                k_left = c.get_grid().dx * c.get_grid().boundary['left'].flux / c.material.thermal_conductivity
                w[0] += r * k_left
            else:
                w[0] += r * val_[0]
            if type(c.get_grid().boundary['right']) == BoundaryConditionFlux:
                k_right = c.get_grid().dx * c.get_grid().boundary['right'].flux / c.material.thermal_conductivity
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
            c.get_grid().val[HALF_STENCIL:c.get_grid().resolution + HALF_STENCIL] += dt * (diffusion[:] +
                                                                                           c.source.y[:] / (c.material.density * c.material.cp))
        else:
            raise TypeError("solver_type should be implicit or explicit")

