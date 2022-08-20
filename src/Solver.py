import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HALF_STENCIL = 1
T0 = 273.15

# TODO: split in a Output class which implements compute_var, and a PostProcess class, which handles the other functions (which are generic).
class Output():

    """Docstring for Output. """

    def __init__(self, index_temporal, var_name='temperature', spatial_type='raw', temporal_type='instantaneous', loc='all'):
        """TODO: to be defined. """

        self.var_name = var_name
        self.spatial_type = spatial_type
        self.temporal_type = temporal_type
        self.loc = loc
        self.index_temporal = index_temporal
        self.temporal_mean = 0.

    def compute_var(self, c):
        if self.var_name == 'temperature':
            if self.loc == 'all':
                return c.y
            else:
                return np.array([c.get_boundary_value(self.loc)])
        elif self.var_name == 'temperature_gradient':
            return np.array([c.get_boundary_gradient(self.loc)])

    def compute_instantaneous(self, c):
        if self.spatial_type == 'raw':
            return self.compute_var(c)[self.index_temporal]
        elif self.spatial_mean == 'spatial_mean':
            return np.mean(self.compute_var(c))
        else:
            raise ValueError

    def compute_temporal(self, c):
        if self.temporal_type == 'temporal_mean':
            raise NotImplemented
        elif self.temporal_type == 'instantaneous':
            return self.compute_instantaneous(c)
        else:
            raise ValueError

    def compute_spatial(self, c):
        return self.compute_var(c)


class Observer:

    """Docstring for Observer. 
    """

    def __init__(self, time_start, time_period, time_end, outputs):
        """TODO: to be defined. """

        # TODO set attributes from component
        assert time_period > 0
        assert time_end > time_start + time_period
        self.time_start = time_start
        self.time_end = time_end
        self.time_period = time_period
        self.outputs = outputs
        self.dt = 0.
        self.ite_start = 0
        self.ite_period = 0
        self.nb_data_per_time = 0
        self.nb_frames = (int)((time_end - time_start) / time_period)
        self.ite_extraction = np.empty((self.nb_frames))
        print("nb frames", self.nb_frames)
        # data = []
        # for i in range(self.nb_frames):
        #     data.append(np.zeros((nb_data_per_time)))
        # self.ts = pd.Series(data, range(self.nb_frames))
        self.update_count = 0
        print("time_period", self.time_period)
        self.temporal_axis = []
        self.temporal = np.zeros((self.nb_frames, len(self.outputs)))

    def set_resolution(self, nb_data_per_time):
        self.nb_data_per_time = nb_data_per_time
        #TODO result becomes attribute of Output. nb_data_per_time is adpated for each output.
        self.result = np.zeros((nb_data_per_time, self.nb_frames, len(self.outputs)))

    def set_frame_ite(self, dt):
        assert self.time_period >= dt
        self.dt = dt
        self.ite_start = (int)(self.time_start / dt)
        self.ite_period = (int)(self.time_period / dt)
        for i in range(self.nb_frames):
            self.ite_extraction[i] = (int)(self.ite_start + i * self.ite_period)
        if self.ite_extraction[0] == 0:
            self.ite_extraction[0] = 1
        print(self.ite_extraction)

    def __get_time(self, ite):
        return self.time_start + ite * self.dt

    def update(self, c, ite):
        print("observer is updated")
        self.temporal_axis.append(self.__get_time(ite))
        for io,output in enumerate(self.outputs):
            self.temporal[self.update_count, io] = output.compute_temporal(c)
        for io,output in enumerate(self.outputs):
            self.result[:, self.update_count, io] = output.compute_spatial(c)
        assert self.update_count < self.nb_frames
        self.update_count += 1

    def is_updated(self, ite):
        return ite in self.ite_extraction

    def plot(self, c):
        fig, ax = plt.subplots()
        resolution = self.result.shape[0]
        x = np.linspace(0, resolution * c.dx, num=resolution)
        for io, output in enumerate(self.outputs):
            for i in range(self.nb_frames):
                time = self.__get_time(self.ite_extraction[i])
                ax.plot(x, self.result[:, i, io], label="t=%ds" % time, linewidth=2.0)
            ax.legend()
            plt.title(f"Component {c.name}, {output.spatial_type} value of {output.var_name}")
            plt.show()

    def plot_temporal(self, c):
        for io, output in enumerate(self.outputs):
            fig, ax = plt.subplots()
            ax.plot(self.temporal_axis, np.array(self.temporal[:,io]), label=f"{output.var_name}[{output.index_temporal}]", linewidth=2.0)
            ax.legend()
            plt.title(f"Component {c.name}, {output.temporal_type} value of {output.var_name}")
            plt.show()


class FiniteVolume:

    """Docstring for FiniteVolume.
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """

    def advance_time(self, dt, component):
        sum_of_fluxes = 0.
        for i, neigh in enumerate(component.neighbours.values()):
            # minus sign because neighbour normal is outward.
            sum_of_fluxes += -component.material.diffusivity * neigh.get_boundary_gradient('in') * neigh.surface
            print('sum of fluxes', sum_of_fluxes)
            print('gradient', neigh.get_boundary_gradient('in'))
        component.y[0] += dt * sum_of_fluxes / component.volume
        print('y', component.y[0])
        # + c.sources[:]


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """

    def update_sources(self, time):
        pass

    def advance_time(self, dt, component):
        dydx = np.diff(component.y)
        diffusion = component.material.diffusivity * (dydx[HALF_STENCIL:component.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / component.dx**2
        component.y[HALF_STENCIL:component.resolution + HALF_STENCIL] += dt * diffusion[:]
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
        for c in self.components:
            if c.observer is not None:
                c.observer.set_frame_ite(self.dt)
            print('dt', self.dt)
            print('dx', c.dx)
            print('diff', c.material.diffusivity)
            if (self.dt / c.dx ** 2) >= 1.0 / (2 * c.material.diffusivity):
                raise ValueError

    def run(self):
        nb_ite = int((self.time_end - self.time_start) / self.dt)
        print("nb_ite", nb_ite)
        for ite in range(1, nb_ite):
            time = self.time_start + ite * self.dt
            for c in self.components:
                c.update()
                # c.update_sources(time)
            for c in self.components:
                c.physics.advance_time(self.dt, c)
            if ite % 10000 == 0:
                print("ite", ite)
                print("time", time)
                for c in self.components:
                    print(c.y)
                    print("")
            for c in self.components:
                if c.observer is not None:
                    if c.observer.is_updated(ite):
                        c.observer.update(c, ite)

    def post(self,):
        for c in self.components:
            if c.observer is not None:
                c.observer.plot(c)
                c.observer.plot_temporal(c)
