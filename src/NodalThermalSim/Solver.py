import copy
import logging
# import mlflow
# from mlflow import log_metric, log_param, log_artifacts
from anytree import Node, RenderTree
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pdas

from NodalThermalSim.Physics import OUTPUT_SIZE, T0, HALF_STENCIL, OutputComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACES = ['left', 'right']
DEFAULT_WKDIR = Path('.')
OUTPUT_FIG_DIR = '.'


def get_time(ite, time_start, dt):
    """Returns the time (s) based on time step and iteration."""

    return time_start + ite * dt


class Output():

    """
    Defines an output. An output consists of a spatial observable and a temporal observable.

    Attributes:
        var_name (str): the name of the variable represented by this class.

        index_temporal (int): index of the grid (numbering starts on left side) at which data
        is extracted for the temporal observables. Used only for spatial_type=instantaneous.
        By default, extraction is at middle index.

        spatial_type (str): type of post-processing for spatial observables: raw or mean.

        temporal_type (str): type of post-processing for temporal observables

        loc (str): location of spatial observable extraction. Can be all, left or right.
        If left or right, the corresponding boundary value is extracted.

    """

    def __init__(self, var_name, index_temporal=-1, spatial_type='raw', loc='all'):

        self.var_name = var_name
        self.loc = loc
        self.index_temporal = index_temporal
        self.spatial_type = spatial_type
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
                assert output.loc == 'left' or output.loc == 'right'
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

    # number of hours above which temporal axis switches to hours instead of seconds.
    NB_HOUR_THRESHOLD_FOR_TEMPORAL_AXIS = 3

    def __init__(self, time_start, time_period, time_end, dt):
        """TODO: to be defined. """

        assert time_period > 0
        assert time_end >= time_start + time_period
        self.time_start = time_start
        self.time_end = time_end
        self.time_period = time_period
        self.dt = dt
        self.nb_frames = (int)((time_end - time_start) / time_period) + 1
        self.ite_extraction = np.empty((self.nb_frames))
        print("Number of data extractions:", self.nb_frames)
        assert self.time_period >= dt
        self.ite_start = (int)(self.time_start / dt)
        self.ite_period = (int)(self.time_period / dt)
        for i in range(self.nb_frames):
            self.ite_extraction[i] = (int)(self.ite_start + i * self.ite_period)
        print("Extraction period (in ite):", (int)(self.time_period / self.dt))
        print('Data are extracted at iterations:', self.ite_extraction)
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
        data = {}
        for io, output in enumerate(c.outputs):
            for i, time in enumerate(self.temporal_axis):
                data[f'{time: .2f}'] = output.result[:,i]
            df = pdas.DataFrame.from_dict(data)
            suffix = ''
            if output.loc in FACES:
               suffix = f'_at_face_{output.loc}'
            df.to_csv(output_dir / f'Component_{c.name}_raw_value_of_{output.var_name}{suffix}.csv', sep=':')
            fig, ax = plt.subplots()
            for i in range(self.nb_frames):
                time = get_time(self.ite_extraction[i], self.time_start, self.dt)
                ax.plot(output.x, output.result[:, i], '-o', label="t=%ds" % time, linewidth=2.0)
            ax.legend()
            plt.title(f"Component {c.name}\n raw value of {output.var_name}{suffix}")
            plt.savefig(output_dir / f"Component_{c.name}_raw_{output.var_name}{suffix}.png")
            plt.close()

    def plot_temporal(self, c, output_dir):
        self.temporal_axis = np.array(self.temporal_axis)
        for io, output in enumerate(c.outputs):
            fig, ax = plt.subplots()
            unit_time = 's'
            if self.temporal_axis[-1] > self.NB_HOUR_THRESHOLD_FOR_TEMPORAL_AXIS * 3600:
                self.temporal_axis /= 3600
                unit_time = 'h'
            ax.plot(self.temporal_axis, np.array(output.temporal_result), '-o', label=f"{output.var_name}", linewidth=2.0)
            ax.legend()
            # ax.xlabel(f'({unit_time})')
            if output.spatial_type == 'raw':
                loc_prefix = 'at loc_'
                if output.loc == 'all':
                    loc = output.index_temporal
                else:
                    loc = output.loc
            else:
                loc_prefix = ''
                loc = ''
            plt.title(f"Component {c.name}\n instantaneous value of\n {output.spatial_type} spatial "
                      f"{output.var_name} {loc_prefix}{loc}")
            plt.savefig(output_dir /
                        f"Component_{c.name}_instantaneous_value_of_{output.spatial_type}_spatial_{output.var_name}_{loc_prefix}{loc}.png")
            plt.close()

class Solver:

    """Docstring for Solver. """

    # number of prints during the time marching.
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
        # check that the non-constant neighbour of a component is the list of components to solve.
        from NodalThermalSim.Component import Component1D, ConstantComponent, Box
        for c in self.components:
            print(f"checked component: {c.name}")
            neighbours_to_check = []
            if type(c) is Component1D:
                neighbours_to_check.append(c.grid.neighbours)
            elif type(c) is Box:
                for grid in c.grid.values():
                    neighbours_to_check.append(grid.neighbours)
            for neighbours in neighbours_to_check:
                for neigh in neighbours.values():
                    if neigh is not None and type(neigh) is not ConstantComponent:
                        if not neigh in self.components:
                            logger.info(f"Component {neigh.name} is a neighbour component of one of the components \
                            to solve, and requires time advance and thus to be included in the components to solve.")
                            # raise ValueError(f"Component {neigh.name} is a neighbour component of one of the components \
                            # to solve, and requires time advance and thus to be included in the components to solve.")
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
            print(f'Component source values ({message}):', c.source.y)
            print(f"Ghost values: left={c.get_grid().get_ghost_value('left')}, "
                  f"right={c.get_grid().get_ghost_value('right')}")
        print('')

    def run(self):
        self.nb_ite = int((self.time_end - self.time_start) / self.dt)
        print('Solver dt:', self.dt)
        print("nb_ite", self.nb_ite)
        intermediate_status_period = max(1, (int)(self.nb_ite / self.NB_STATUS))
        # mlflow.log_param('nb_ite', self.nb_ite)
        # mlflow.log_param('timestep', self.dt)
        for ite in range(0, self.nb_ite+1):
            time = get_time(ite, self.time_start, self.dt)
            # mlflow.log_metric('iteration', ite)
            # mlflow.log_metric('time', time)
            for c in self.components:
                c.update_ghost_node(time, ite)
                if self.observer is not None:
                    if self.observer.is_updated(ite):
                        self.observer.update_components(c, self.post)
            if ite % intermediate_status_period == 0:
                self.show_status(ite, time)
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
