import matplotlib.pyplot as plt
import numpy as np


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
        diffusion = c.material.diffusivity * (dydx[1:c.resolution + 1] - dydx[:-1]) / c.dx**2
        # print('y', self.y)
        # print('diffusion', diffusion)
        return diffusion

    def update_sources(self, time):
        pass

    def advance_time(self, c):
        c.y[1:c.resolution + 1] += self.dt * self.compute_diffusion(c)[:]
        # + c.sources[:]


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
            if (self.dt / c.dx ** 2) >= 1.0 / (2 * c.material.diffusivity):
                raise ValueError
        nb_ite = int((self.time_end - self.time_start) / self.dt)
        print("nb_ite", nb_ite)
        for ite in range(1, nb_ite):
            time = self.time_start + ite * self.dt
            for c in self.components:
                c.update()
                c.update_sources(time)
            for c in self.components:
                self.fd_transport.advance_time(c)
            if ite % 1 == 0:
                print("ite", ite)
                print("time", time)
                for c in self.components:
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
