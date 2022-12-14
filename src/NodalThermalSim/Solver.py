import copy
import logging
# import mlflow
# from mlflow import log_metric, log_param, log_artifacts
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# import pandas as pd
from NodalThermalSim.Physics import OUTPUT_SIZE, T0, HALF_STENCIL, OutputComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_WKDIR = Path('.')
OUTPUT_FIG_DIR = 'Results'


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
        self.temporal_result = np.empty((1))
        self.result = np.empty((1,1))


class Post():

    """Transform the Output according to the type of spatial and temporal post-processing. """

    def __init__(self):
        self.temporal_mean = 0.
        self.output_computer = OutputComputer()

    def set_size(self, c, nb_frames, output):
        if output.spatial_type == 'raw':
            if output.loc == 'all':
                output.size = OUTPUT_SIZE(output.var_name, c.get_grid().resolution)
                output.x = c.get_grid().get_physics_x()[:output.size]
                # default index for temporal value output ist the middle of the component.
                if output.index_temporal == -1:
                    output.index_temporal = (int)(output.size/2)
            else:
                output.size = 1
                output.x = np.array([c.get_grid().x[c.get_grid().BOUNDARY_VAL_INDEX[output.loc]]])
        elif output.spatial_type == 'mean':
            output.size = 1
            # mean value is located at the middle of the component.
            output.x = np.array([c.get_grid().get_physics_x()[(int)(c.get_grid().resolution/2)]])
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

    def set_output_container(self, post, c):
        for o in c.outputs:
            post.set_size(c, self.nb_frames, o)
            o.temporal_result = np.resize(o.temporal_result, (self.nb_frames))

    def update(self, ite, post):
        print("Observer is updated")
        self.temporal_axis.append(get_time(ite, self.time_start, self.dt))
        assert self.update_count < self.nb_frames
        self.update_count += 1

    def update_components(self, c, post):
        for io, output in enumerate(c.outputs):
            output.temporal_result[self.update_count] = post.compute_instantaneous(c, output)
            output.result[:, self.update_count] = post.compute_spatial(c, output)

    def is_updated(self, ite):
        return ite in self.ite_extraction

    def plot(self, c, output_dir):
        for io, output in enumerate(c.outputs):
            fig, ax = plt.subplots()
            for i in range(self.nb_frames):
                time = get_time(self.ite_extraction[i], self.time_start, self.dt)
                ax.plot(output.x, output.result[:, i], '-o', label="t=%ds" % time, linewidth=2.0)
            ax.legend()
            plt.title(f"Component {c.name}\n raw value of {output.var_name}")
            plt.savefig(output_dir / f"Component_{c.name}_raw_{output.var_name}.png")
            plt.close()

    def plot_temporal(self, c, output_dir):
        for io, output in enumerate(c.outputs):
            fig, ax = plt.subplots()
            ax.plot(self.temporal_axis, np.array(output.temporal_result), '-o', label=f"{output.var_name}", linewidth=2.0)
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
            plt.title(f"Component {c.name}\n {output.temporal_type} value of\n {output.spatial_type} spatial "
                      f"{output.var_name} {loc_prefix}{loc}")
            # loc should be in meter or in percentage of thickness
            plt.savefig(output_dir /
                        f"Component_{c.name}_{output.temporal_type}_of_{output.spatial_type}_spatial_{output.var_name}_{loc_prefix}{loc}.png")
            plt.close()

class Solver:

    """Docstring for Solver. """

    NB_STATUS = 10

    def __init__(self, component_list, dt, time_end, observer, time_start = 0., solver_type='implicit'):
        self.solver_type = solver_type
        self.components = component_list
        # TODO: pass this dt to the equation time advance. Remove dt attribute in equation.
        self.dt = dt
        self.time_start = time_start
        self.time_end = time_end
        self.nb_ite = -1
        self.post = Post()
        self.observer = observer
        self.node = Node('')
        for c in self.components:
            c.check()
            if self.solver_type == 'explicit':
                c.check_stability(dt)
            self.observer.set_output_container(self.post, c)
            c.add_to_tree(self.node)


    def show_tree(self):
        print("\n")
        for pre, fill, node_ in RenderTree(self.node):
            print("%s%s" % (pre, node_.name))

    def show_status(self, ite, time):
        print(f"Current ite {ite} on {self.nb_ite}")
        print(f"Current time {time} on {self.time_end - self.time_start}")
        for c in self.components:
            logger.log(logging.INFO, f'Name {c.name}')
            print(f'Name {c.name}')
            if c.get_grid().resolution < 100:
                message = 'all values'
                phys_values = c.get_grid().get_physics_val()
            else:
                message = 'first 100 values'
                phys_values = c.get_grid().get_physics_val()[:100]
            print(f'Component physical values ({message}):', phys_values)
            print(f"Actual ghost values: left={c.get_grid().get_ghost_value('left')}, "
                  f"right={c.get_grid().get_ghost_value('right')}")
            print(f"Target ghost values: left={c.get_grid().ghost_target_val['left']}, "
                  f"right={c.get_grid().ghost_target_val['right']}")
            print("--")
        print('')

    def run(self):
        self.nb_ite = int((self.time_end - self.time_start) / self.dt)
        print('Solver dt:', self.dt)
        print("nb_ite", self.nb_ite)
        intermediate_status_period = max(1, (int)(self.nb_ite / self.NB_STATUS))
        # mlflow.log_param('nb_ite', self.nb_ite)
        # mlflow.log_param('timestep', self.dt)
        for ite in range(0, self.nb_ite):
            time = get_time(ite, self.time_start, self.dt)
            # mlflow.log_metric('iteration', ite)
            # mlflow.log_metric('time', time)
            if ite % intermediate_status_period == 0:
                self.show_status(ite, time)
            for c in self.components:
                c.update_ghost_node(time, ite)
                if self.observer is not None:
                    if self.observer.is_updated(ite):
                        self.observer.update_components(c, self.post)
            for c in self.components:
                c.physics.advance_time(self.dt, c, ite, self.solver_type)
            # TODO put is_updated in update function. put component loop in update function.
            if self.observer is not None:
                if self.observer.is_updated(ite):
                    self.observer.update(ite, self.post)

    # TODO add a show and save args
    def visualize(self, output_root_dir=DEFAULT_WKDIR):
        output_dir = output_root_dir / OUTPUT_FIG_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        for c in self.components:
            if self.observer is not None:
                self.observer.plot(c, output_dir)
                self.observer.plot_temporal(c, output_dir)
        # mlflow.log_artifacts(output_dir)
