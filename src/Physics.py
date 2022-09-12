import numpy as np
from Solver import HALF_STENCIL, T0

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
        y_mean = np.mean(c.get_physics_y())
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
        dydx = np.diff(c.y)
        diffusion = c.material.thermal_conductivty * (dydx[HALF_STENCIL:c.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / c.dx**2
        c.y[HALF_STENCIL:c.resolution + HALF_STENCIL] += (1. / (c.material.density * c.material.cp)) * dt * (diffusion[:] + c.source.y[:])

