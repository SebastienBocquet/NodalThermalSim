import numpy as np


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
                return c.get_physics_val()
            else:
                return np.array([c.get_boundary_value(output.loc)])
        elif output.var_name == 'temperature_gradient':
            if output.loc == 'all':
                return np.diff(c.get_physics_val()) / c.dx
            else:
                return np.array([c.get_boundary_gradient(output.loc)])
        elif output.var_name == 'heat_flux':
            if output.loc == 'all':
                return c.material.thermal_conductivty * np.diff(c.get_physics_val() / c.dx)
            else:
                return np.array([c.material.thermal_conductivty * c.get_boundary_gradient(output.loc)])
        elif output.var_name == 'HTC':
            surface_temperature = 0.
            ref_temperature = 0.
            if output.loc == 'in':
                # TODO introduce a get_ghost_value()[output.loc]
                surface_temperature = 0.5 * (c.y[0] + c.get_physics_val()[0])
                ref_temperature = c.get_physics_val()[0]
            else:
                # TODO introduce a get_ghost_value()[output.loc]
                surface_temperature = 0.5 * (c.get_physics_val()[-1] + c.y[c.resolution + 1])
                ref_temperature = c.get_physics_val()[-1]
            htc = c.material.thermal_conductivty * c.get_boundary_gradient(output.loc) / (surface_temperature - ref_temperature)
            return np.array([htc])


class FiniteVolume:

    """Docstring for FiniteVolume.
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """
        self.y_mean_conservative = T0
        self.k = 0.01

    def advance_time(self, dt, c, ite):
        dydx = np.diff(c.y)
        diffusion = c.material.diffusivity * (dydx[HALF_STENCIL:c.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / c.dx**2
        c.y[HALF_STENCIL:c.resolution + HALF_STENCIL] += dt * (diffusion[:] + c.source.y[:])
        flux_in = c.material.diffusivity * c.get_boundary_gradient('left') * c.surface
        flux_ext = c.material.diffusivity * c.get_boundary_gradient('right') * c.surface
        temp_variation = (flux_in + flux_ext) / c.volume
        self.y_mean_conservative += temp_variation * dt
        y_mean = np.mean(c.get_physics_val())
        c.y[HALF_STENCIL:c.resolution + HALF_STENCIL] -= self.k * (y_mean - self.y_mean_conservative)

        if ite % 10000 == 0:
            print('y mean by energy conservation', self.y_mean_conservative)
            print('effective y mean', y_mean)


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """

        self.y0 = T0

    def advance_time(self, dt, c, ite):
        dydx = np.diff(c.val)
        diffusion = c.material.thermal_conductivty * (dydx[HALF_STENCIL:c.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / c.dx**2
        c.val[HALF_STENCIL:c.resolution + HALF_STENCIL] += (1. / (c.material.density * c.material.cp)) * dt * (diffusion[:] + c.source.y[:])

