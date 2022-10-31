import copy
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from Physics import OUTPUT_SIZE, T0, HALF_STENCIL, OutputComputer


OUTPUT_FIG = pathlib.Path('Results')
INTERMEDIATE_STATUS_PERIOD = 10000


def get_time(ite, time_start, dt):
    """Returns the time (s) based on time step and iteration."""

    return time_start + ite * dt


class Output():

    """
    Defines an output.

    Attributes:
        var_name (str): the name of the variable represented by this class.

    """

    def __init__(self, var_name, index_temporal=-1, spatial_type='raw', temporal_type='instantaneous', loc='all'):

        self.var_name = var_name
        self.loc = loc
        self.index_temporal = index_temporal
        self.spatial_type = spatial_type
        self.temporal_type = temporal_type
        self.x = np.empty((1))
        self.result = np.empty((1,1))


class Post():

    """Transform the Output according to the type of spatial and temporal post-processing. """

    def __init__(self):
        self.temporal_mean = 0.
        self.output_computer = OutputComputer()

    def set_size(self, c, nb_frames, output):
        # TODO define x in component. Start at ghost node.
        x = np.linspace(0, c.resolution * c.dx, num=c.resolution)
        if output.spatial_type == 'raw':
            if output.loc == 'all':
                output.size = OUTPUT_SIZE(output.var_name, c.resolution)
                output.x = x[:output.size]
                if output.index_temporal == -1:
                    output.index_temporal = (int)(output.size/2)
            else:
                output.size = 1
                output.x = np.array(x[c.boundary_val_index[output.loc]])
        elif output.spatial_type == 'mean':
            output.size = 1
            output.x = np.array(x[(int)(c.resolution/2)])
        else:
            raise ValueError
        output.result = np.resize(output.result, (output.size, nb_frames))


    def compute_instantaneous(self, c, output):
        if output.spatial_type == 'raw':
            output_data = self.output_computer.compute_var(c, output)
            if len(output_data) == 1:
                return output_data[0]
            else:
                return output_data[output.index_temporal]
        elif output.spatial_type == 'mean':
            output_data = self.output_computer.compute_var(c, output)
            return np.array([np.mean(output_data)])
        else:
            raise ValueError

    def compute_temporal(self, c, output):
        if output.temporal_type == 'mean':
            raise NotImplemented
        elif output.temporal_type == 'instantaneous':
            return self.compute_instantaneous(c, output)
        else:
            raise ValueError

    def compute_spatial(self, c, output):
        if output.spatial_type == 'raw':
            return self.output_computer.compute_var(c, output)
        elif output.spatial_type == 'mean':
            output_data = self.output_computer.compute_var(c, output)
            return np.array([np.mean(output_data)])
        else:
            raise ValueError

class Observer:

    """Docstring for Observer. 
    """

    def __init__(self, time_start, time_period, time_end, dt):
        """TODO: to be defined. """

        assert time_period > 0
        assert time_end >= time_start + time_period
        self.time_start = time_start
        self.time_end = time_end
        self.time_period = time_period
        self.dt = dt
        self.nb_frames = (int)((time_end - time_start) / time_period)
        self.ite_extraction = np.empty((self.nb_frames))
        print("Number of data extractions:", self.nb_frames)
        assert self.time_period >= dt
        self.ite_start = (int)(self.time_start / dt)
        self.ite_period = (int)(self.time_period / dt)
        for i in range(self.nb_frames):
            self.ite_extraction[i] = (int)(self.ite_start + i * self.ite_period)
        print("Extraction period (in ite):", (int)(self.time_period / self.dt))
        print('Data are extracted at iterations:', self.ite_extraction)
        # data = []
        # for i in range(self.nb_frames):
        #     data.append(np.zeros((nb_data_per_time)))
        # self.ts = pd.Series(data, range(self.nb_frames))
        self.update_count = 0
        self.temporal_axis = []
        OUTPUT_FIG.mkdir(parents=True, exist_ok=True)

    def set_output_container(self, post, c):
        for o in c.outputs:
            post.set_size(c, self.nb_frames, o)

    def update(self, c, ite, post):
        print("Observer is updated")
        self.temporal_axis.append(get_time(ite, self.time_start, self.dt))
        for io, output in enumerate(c.outputs):
            c.temporal_output[self.update_count, io] = post.compute_temporal(c, output)
            output.result[:, self.update_count] = post.compute_spatial(c, output)
        assert self.update_count < self.nb_frames
        self.update_count += 1

    def is_updated(self, ite):
        return ite in self.ite_extraction

    def plot(self, c):
        for io, output in enumerate(c.outputs):
            if output.size > 1:
                fig, ax = plt.subplots()
                for i in range(self.nb_frames):
                    time = get_time(self.ite_extraction[i], self.time_start, self.dt)
                    ax.plot(output.x, output.result[:, i], '-o', label="t=%ds" % time, linewidth=2.0)
                ax.legend()
                plt.title(f"Component {c.name}\n raw value of {output.var_name}")
                plt.savefig(OUTPUT_FIG / f"Component_{c.name}_raw_{output.var_name}.png")
                plt.close()
                # plt.show()

    def plot_temporal(self, c):
        for io, output in enumerate(c.outputs):
            fig, ax = plt.subplots()
            ax.plot(self.temporal_axis, np.array(c.temporal_output[:,io]), label=f"{output.var_name}", linewidth=2.0)
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
            plt.title(f"Component {c.name}\n {output.temporal_type} value of\n {output.spatial_type} spatial {output.var_name} {loc_prefix}{loc}")
            plt.savefig(OUTPUT_FIG / f"Component_{c.name}_{output.temporal_type}_of_{output.spatial_type}_spatial_{output.var_name}_{loc_prefix}{loc}.png")
            plt.close()
            # plt.show()


class Solver:

    """Docstring for Solver. """

    def __init__(self, component_list, dt, time_end, observer, time_start = 0.):
        """TODO: to be defined. """
        self.components = component_list
        # TODO: pass this dt to the equation time advance. Remove dt attribute in equation.
        self.dt = dt
        self.time_start = time_start
        self.time_end = time_end
        self.post = Post()
        self.observer = observer
        self.node = Node('')
        print('Solver dt:', self.dt)
        for c in self.components:
            c.check()
            c.check_stability(dt)
            c.add_to_tree(self.node)
            c.set_temporal_data_size(self.observer.nb_frames)
            self.observer.set_output_container(self.post, c)


    def show_tree(self):
        print("\n")
        for pre, fill, node_ in RenderTree(self.node):
            print("%s%s" % (pre, node_.name))

    def show_status(self, ite, time):
        print("ite", ite)
        print("time", time)
        for c in self.components:
            if c.resolution < 100:
                message = 'all values'
                phys_values = c.get_physics_y()
            else:
                message = 'first 100 values'
                phys_values = c.get_physics_y()[:100]
            print('Component name:', c.name)
            print(f'Component physical values ({message}):', phys_values)
            print("")

    def run(self):
        nb_ite = int((self.time_end - self.time_start) / self.dt)
        print("nb_ite", nb_ite)
        for ite in range(0, nb_ite):
            time = get_time(ite, self.time_start, self.dt)
            for c in self.components:
                if self.observer is not None:
                    if self.observer.is_updated(ite):
                        self.observer.update(c, ite, self.post)
            if ite % INTERMEDIATE_STATUS_PERIOD == 0:
                self.show_status(ite, time)
            for c in self.components:
                c.update(time, ite)
            for c in self.components:
                c.physics.advance_time(self.dt, c, ite)

    def compute_post(self,):
        for c in self.components:
            if self.observer is not None:
                self.observer.plot(c)
                self.observer.plot_temporal(c)
