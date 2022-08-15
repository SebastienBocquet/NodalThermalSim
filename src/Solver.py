import matplotlib.pyplot as plt
import numpy as np

HALF_STENCIL = 1

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


class FiniteVolume:

    """Docstring for FiniteVolume.
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self, dt):
        """TODO: to be defined. """
        self.dt = dt

    def advance_time(self, component):
        sum_of_fluxes = 0.
        for neigh in component.neighbours.values():
            # minus sign because neighbour normal is outward.
            sum_of_fluxes += -component.material.diffusivity * neigh.get_boundary_gradient('in') * neigh.surface
        component.y += sum_of_fluxes / component.volume
        # + c.sources[:]


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self, dt):
        """TODO: to be defined. """
        self.dt = dt

    def update_sources(self, time):
        pass

    def advance_time(self, component):
        dydx = np.diff(component.y)
        diffusion = component.material.diffusivity * (dydx[HALF_STENCIL:component.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / component.dx**2
        component.y[HALF_STENCIL:component.resolution + HALF_STENCIL] += self.dt * diffusion[:]
        # + c.sources[:]


# TODO equation becomes an attribute of the component
class Solver:

    """Docstring for Solver. """

    def __init__(self, component_list, dt, time_end, time_start = 0.):
        """TODO: to be defined. """
        self.components = component_list
        # TODO: pass this dt to the equation time advance. Remove dt attribute in equation.
        self.dt = dt
        self.time_start = time_start
        self.time_end = time_end

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
                # c.update_sources(time)
            for c in self.components:
                c.physics.advance_time(c)
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
                        c.get_boundary_gradient('in')
                        c.get_boundary_gradient('ext')

    def post(self,):
        for c in self.components:
            if c.observer is not None:
                c.observer.plot(c.dx)
