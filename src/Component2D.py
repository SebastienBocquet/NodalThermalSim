import matplotlib.pyplot as plt
import numpy as np


class Air:

    """Docstring for Air. """

    def __init__(self, name, temperature):
        """TODO: to be defined. """

        self.name = name
        self.y = temperature

    def get_bc_val_in(self):
        return self.y

    def get_bc_val_ext(self):
        return self.y


class Observer:

    """Docstring for Observer. 
    """

    def __init__(self, time_start, time_period, time_end, resolution, dt):
        """TODO: to be defined. """

        assert time_period >= dt
        assert time_period > 0
        assert time_end > time_start + time_period
        self.time_start = time_start
        self.time_end = time_end
        self.time_period = time_period
        self.dt = dt
        self.nb_frames = (int)((time_end - time_start) / time_period)
        print("nb frames", self.nb_frames)
        self.result = np.zeros((resolution + 2, self.nb_frames))
        self.update_count = 0
        self.ite_start = (int)(self.time_start / dt)
        print("time_period", self.time_period)
        print("dt", dt)
        self.ite_period = (int)(self.time_period / dt)

    def update(self, y, i):
        print("observer is updated")
        assert self.update_count < self.nb_frames
        self.result[:, self.update_count] = y[:]
        self.update_count += 1

    def is_updated(self, ite):
        print("observer ite", ite)
        print("observer ite start", self.ite_start)
        print("observer ite_period", self.ite_period)
        is_updated = (ite - self.ite_start) % self.ite_period == 0
        print("is updated", is_updated)
        return is_updated

    def plot(self, dx):
        fig, ax = plt.subplots()
        resolution = self.result.shape[0]
        x = np.linspace(0, resolution * dx, num=resolution)
        for i in range(self.nb_frames):
            time = self.time_start + i * self.dt
            ax.plot(x, self.result[:, i], label="t=%ds" % time, linewidth=2.0)
        ax.legend()
        plt.show()


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self, dt):
        """TODO: to be defined. """
        self.dt = dt

    def compute_diffusion(self, c):
        dydx = np.diff(c.y)
        diffusion = c.diffusivity * (dydx[1:c.resolution + 1] - dydx[:-1]) / c.dx**2
        # print('y', self.y)
        # print('diffusion', diffusion)
        return diffusion

    def update_sources(self, time):
        pass

    def advance_time(self, c):
        c.y[1:c.resolution + 1] += self.dt * self.compute_diffusion(c)[:]
        # + c.sources[:]



class Component2D:

    """Docstring for Component2D. 
    in           ext
    y[0]-------->y[resolution-1]
    """

    def __init__(
        self,
        name,
        cp,
        density,
        thermal_conductivty,
        thickness,
        y0,
        component_in_neighbour,
        component_ext_neighbour,
        observer=None,
        resolution=10,
        surface=1.0,
    ):
        """TODO: to be defined. """

        self.name = ""
        self.cp = cp
        self.density = density
        self.thermal_conductivty = thermal_conductivty
        self.resolution = resolution
        self.component_in_neighbour = component_in_neighbour
        self.component_ext_neighbour = component_ext_neighbour
        self.thickness = thickness
        self.surface = surface
        self.y = np.zeros((self.resolution + 2)) + y0
        self.sources = np.zeros((self.resolution))
        self.dx = self.thickness / self.resolution
        self.diffusivity = thermal_conductivty / (density * cp)
        self.observer = observer
        if observer is not None:
            assert self.observer.result.shape[0] == self.resolution + 2

    def compute_diffusion(self):
        coef = (self.thermal_conductivty / (self.density * self.cp)) * (
            1.0 / self.dx ** 2
        )
        dydx = np.diff(self.y)
        diffusion = dydx[1:self.resolution + 1] - dydx[:-1]
        diffusion *= coef
        # print('y', self.y)
        # print('diffusion', diffusion)
        return diffusion

    def compute_wall_heat_flux(self, y):
        assert len(y) == 2
        return -self.thermal_conductivty * (y[1] - y[0]) / self.dx

    def get_wall_heat_flux_in(self,):
        y_border = self.y[:2]
        wall_heat_flux_in = self.compute_wall_heat_flux(y_border)
        print('wall flux in', wall_heat_flux_in)
        return wall_heat_flux_in

    def get_wall_heat_flux_ext(self,):
        y_reverse = self.y[::-1]
        y_border = y_reverse[:2]
        wall_heat_flux_ext = self.compute_wall_heat_flux(y_border)
        print('wall flux in', wall_heat_flux_ext)
        return wall_heat_flux_ext

    def get_bc_val_in(self):
        return self.y[1]

    def get_bc_val_ext(self):
        return self.y[self.resolution]

    def update_bc(self):
        self.y[0] = self.component_in_neighbour.get_bc_val_ext()
        self.y[self.resolution + 1] = self.component_ext_neighbour.get_bc_val_in()

    def update_sources(self, time):
        pass


class Solver:

    """Docstring for Solver. """

    def __init__(self, component_list, fd_transport, dt, time_end, time_start = 0):
        """TODO: to be defined. """
        self.components = component_list
        self.dt = dt
        self.time_start = time_start
        self.time_end = time_end
        self.fd_transport = fd_transport

    def run(self):
        for c in self.components:
            diffusivity = c.thermal_conductivty / (c.density * c.cp)
            if (self.dt / c.dx ** 2) >= 1.0 / (2 * diffusivity):
                raise ValueError
        nb_ite = int((self.time_end - self.time_start) / self.dt)
        print("nb_ite", nb_ite)
        for ite in range(1, nb_ite):
            time = self.time_start + ite * self.dt
            for c in self.components:
                c.update_bc()
                c.update_sources(time)
            for c in self.components:
                self.fd_transport.advance_time(c)
            if ite % 1 == 0:
                print("ite", ite)
                print("time", time)
                for c in self.components:
                    print(c.name)
                    print(c.y)
                    print("")
            for c in self.components:
                if c.observer is not None:
                    if c.observer.is_updated(ite):
                        c.observer.update(c.y, ite)
                        c.get_wall_heat_flux_in()
                        c.get_wall_heat_flux_ext()

    def post(self,):
        for c in self.components:
            if c.observer is not None:
                c.observer.plot(c.dx)
