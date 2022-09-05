import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd

# 0 means the output size is the component resolution.
# otherwise the size is 1
OUTPUT_SIZE = {'temperature': 0, 'temperature_gradient': 1, 'HTC': 1}

HALF_STENCIL = 1
T0 = 273.15
OUTPUT_FIG = pathlib.Path('Results')

# TODO: split in a Output class which implements compute_var, and a PostProcess class, which handles the other functions (which are generic).
class Output():

    """Docstring for Output. """

    def __init__(self, index_temporal, var_name, spatial_type='raw', temporal_type='instantaneous', loc='all'):
        """TODO: to be defined. """

        self.var_name = var_name
        self.spatial_type = spatial_type
        self.temporal_type = temporal_type
        self.loc = loc
        self.index_temporal = index_temporal
        self.temporal_mean = 0.
        pathlib.Path(OUTPUT_FIG).mkdir(parents=True, exist_ok=True)

    def set_size(self, c):
        x = np.linspace(0, c.resolution * c.dx, num=c.resolution)
        if OUTPUT_SIZE[self.var_name] == 0:
            self.size = c.resolution
            self.x = x
        elif OUTPUT_SIZE[self.var_name] == 1:
            self.size = 1
            self.x = np.array(x[c.boundary_loc[self.loc]])
        else:
            raise ValueError

    def compute_var(self, c):
        if self.var_name == 'temperature':
            if self.loc == 'all':
                return c.get_physics_y()
            else:
                return np.array([c.get_boundary_value(self.loc)])
        elif self.var_name == 'temperature_gradient':
            return np.array([c.get_boundary_gradient(self.loc)])
        elif self.var_name == 'HTC':
            surface_temperature = 0.
            ref_temperature = 0.
            if self.loc == 'in':
                # TODO introduce a get_ghost_value()[self.loc]
                surface_temperature = 0.5 * (c.y[0] + c.get_physics_y()[0])
                ref_temperature = c.get_physics_y()[0]
            else:
                # TODO introduce a get_ghost_value()[self.loc]
                surface_temperature = 0.5 * (c.get_physics_y()[-1] + c.y[c.resolution + 1])
                ref_temperature = c.get_physics_y()[-1]
            htc = c.material.thermal_conductivty * c.get_boundary_gradient(self.loc) / (surface_temperature - ref_temperature)
            return np.array([htc])


    def compute_instantaneous(self, c):
        if self.spatial_type == 'raw':
            if len(self.compute_var(c)) == 1:
                return self.compute_var(c)[0]
            else:
                return self.compute_var(c)[self.index_temporal]
        elif self.spatial_type == 'mean':
            return np.array([np.mean(self.compute_var(c))])
        else:
            raise ValueError

    def compute_temporal(self, c):
        if self.temporal_type == 'mean':
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
        self.nb_frames = (int)((time_end - time_start) / time_period)
        self.ite_extraction = np.empty((self.nb_frames))
        print("nb frames", self.nb_frames)
        # data = []
        # for i in range(self.nb_frames):
        #     data.append(np.zeros((nb_data_per_time)))
        # self.ts = pd.Series(data, range(self.nb_frames))
        self.update_count = 0
        print("time period", self.time_period)
        self.temporal_axis = []
        self.temporal = np.zeros((self.nb_frames, len(self.outputs)))

    # add a set_output func.

    def set_output_container(self, c):
        for o in self.outputs:
            o.set_size(c)
            o.result = np.zeros((o.size, self.nb_frames))

    def set_frame_ite(self, dt):
        assert self.time_period >= dt
        self.dt = dt
        print("nb ite period", (int)(self.time_period / self.dt))
        self.ite_start = (int)(self.time_start / dt)
        self.ite_period = (int)(self.time_period / dt)
        for i in range(self.nb_frames):
            self.ite_extraction[i] = (int)(self.ite_start + i * self.ite_period)
        if self.ite_extraction[0] == 0:
            self.ite_extraction[0] = 1
        print('data extract at ite', self.ite_extraction)

    def __get_time(self, ite):
        return self.time_start + ite * self.dt

    def update(self, c, ite):
        print("observer is updated")
        self.temporal_axis.append(self.__get_time(ite))
        for io, output in enumerate(self.outputs):
            self.temporal[self.update_count, io] = output.compute_temporal(c)
        for io, output in enumerate(self.outputs):
            output.result[:, self.update_count] = output.compute_spatial(c)
        assert self.update_count < self.nb_frames
        self.update_count += 1

    def is_updated(self, ite):
        return ite in self.ite_extraction

    def plot(self, c):
        for io, output in enumerate(self.outputs):
            if output.size > 1:
                fig, ax = plt.subplots()
                for i in range(self.nb_frames):
                    time = self.__get_time(self.ite_extraction[i])
                    ax.plot(output.x, output.result[:, i], '-o', label="t=%ds" % time, linewidth=2.0)
                ax.legend()
                plt.title(f"Component {c.name}, raw value of {output.var_name}")
                plt.savefig(OUTPUT_FIG / f"Component_{c.name}_raw_{output.var_name}.png")
                # plt.show()

    def plot_temporal(self, c):
        for io, output in enumerate(self.outputs):
            fig, ax = plt.subplots()
            ax.plot(self.temporal_axis, np.array(self.temporal[:,io]), label=f"{output.var_name}", linewidth=2.0)
            ax.legend()
            loc = 0
            loc_prefix = ''
            if output.spatial_type == 'raw':
                loc_prefix = 'at loc_'
                if output.loc == 'all':
                    loc = output.index_temporal
                else:
                    loc = output.loc
            else:
                loc_prefix = ''
                loc = ''
            plt.title(f"Component {c.name}, {output.temporal_type} value of {output.spatial_type} spatial {output.var_name} {loc_prefix}{loc}")
            plt.savefig(OUTPUT_FIG / f"Component_{c.name}_{output.temporal_type}_of_{output.spatial_type}_spatial_{output.var_name}_{loc_prefix}{loc}.png")
            # plt.show()


class FiniteVolume:

    """Docstring for FiniteVolume.
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """
        self.delta_temperature = 0.

    def advance_time(self, dt, component, ite):
        dydx = np.diff(component.y)
        diffusion = component.material.diffusivity * (dydx[HALF_STENCIL:component.resolution + HALF_STENCIL] - dydx[:-HALF_STENCIL]) / component.dx**2
        component.y[HALF_STENCIL:component.resolution + HALF_STENCIL] += dt * diffusion[:]
        sum_of_fluxes = 0.
        flux_in = component.material.diffusivity * component.get_boundary_gradient('ext') * component.surface
        flux_ext = component.material.diffusivity * component.get_boundary_gradient('in') * component.surface
        temp_variation = (flux_in + flux_ext) / component.volume
        self.delta_temperature += temp_variation * dt
        if ite % 10000 == 0:
            print('delta T mean', self.delta_temperature)
            # print('sum of fluxes', sum_of_fluxes)
            # print('gradient', neigh.get_boundary_gradient('in'))
        # component.y[1] += dt * sum_of_fluxes / component.volume
        # print('y', component.y[1])
        # + c.sources[:]


class FiniteDifferenceTransport:

    """Docstring for FiniteDifferenceTransport. 
    """

    # y0, y, dx belong to Component. Pass them to FDTransport (or pass the entire Component)
    def __init__(self):
        """TODO: to be defined. """

    def update_sources(self, time):
        pass

    def advance_time(self, dt, component, ite):
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
            print('diffusivity', c.material.diffusivity)
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
                c.physics.advance_time(self.dt, c, ite)
            if ite % 10000 == 0:
                print("ite", ite)
                print("time", time)
                for c in self.components:
                    print(c.get_physics_y())
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
