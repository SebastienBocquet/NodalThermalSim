import sys
import os
import numpy as np
from pytest import approx
from NodalThermalSim.Component import ConstantComponent, Material, Component1D
from NodalThermalSim.Solver import Solver, Observer, Output
from NodalThermalSim.Physics import FiniteDifferenceTransport

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE)
CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100
air = Material(CP_AIR, DENSITY_AIR, K_AIR)

# build a room component, as a 1D component of thickness BOX_WIDTH.
RESOLUTION = 10
BOX_DEPTH = 1.
BOX_WIDTH = 2.
BOX_HEIGHT = 2.5
DX = BOX_WIDTH / (RESOLUTION - 1)
DT = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))

TIME_START = 0.
TIME_END = 1 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

X_PHYSICS = np.linspace(0, (RESOLUTION - 1) * DX, num=RESOLUTION)
# constructed such as left gradient is T0 / dx
INIT_AIR_TEMPERATURE = np.linspace(2 * T0, 2 * T0 + (RESOLUTION - 1) * T0, num=RESOLUTION)

neighbours = {'left': air_interior, 'right': air_exterior}

INDEX_TEMPORAL = 0
INDEX_TEMPORAL_DEFAULT = (int)(0.5 * RESOLUTION)
output_temperature = Output('temperature')
output_gradient = Output('temperature_gradient')
output_temperature_temporal_loc = Output('temperature', INDEX_TEMPORAL)
output_temperature_space_avg = Output(var_name='temperature', spatial_type='mean')
output_gradient_left = Output('temperature_gradient', loc='left')
output_gradient_right = Output('temperature_gradient', loc='right')


def test_observer():
    observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature],
                       resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)

    # TODO check extreme setup. Especially the case of a single frame (typically the last one).
    observed_ite = []
    for i in range(NB_FRAMES):
        ite_observation = (int)(i * (int)(OBSERVER_PERIOD / DT))
        observed_ite.append(ite_observation)
    assert (observed_ite == observer.ite_extraction).all()

    for i in range(NB_FRAMES):
        ite_observation = (int)(i * (int)(OBSERVER_PERIOD / DT))
        observed_ite.append(ite_observation)
        print(ite_observation)
        assert observer.is_updated(ite_observation) is True

    component_to_solve_list = [room]
    solver = Solver(component_to_solve_list, DT, TIME_END, observer)
    solver.run()
    assert observer.update_count == NB_FRAMES


def test_observer_ite0():
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    # add a set_observer function to Component
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)

    # TODO check extreme setup. Especially the case of a single frame (typically the last one).
    expected_observed_ite = 0
    observed_ite = [expected_observed_ite]
    print(observer.ite_extraction)
    assert (observed_ite == observer.ite_extraction).all()
    assert observer.is_updated(expected_observed_ite) is True

    component_to_solve_list = [room]
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    assert observer.update_count == 1


def test_raw_output():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration.
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    assert output_temperature.size == RESOLUTION
    assert output_temperature.x == approx(X_PHYSICS)
    assert (output_temperature.result[:,0] == INIT_AIR_TEMPERATURE).all()
    assert len(observer.temporal_axis) == 1
    assert observer.temporal_axis[:] == [TIME_START]
    assert output_temperature.temporal_result[0] == INIT_AIR_TEMPERATURE[INDEX_TEMPORAL_DEFAULT]


def test_raw_output_temporal_loc():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature_temporal_loc], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration.
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    assert output_temperature_temporal_loc.temporal_result[0] == INIT_AIR_TEMPERATURE[INDEX_TEMPORAL]


def test_spatial_avg_output():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature_space_avg], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    assert output_temperature_space_avg.size == 1
    assert (output_temperature_space_avg.x == [X_PHYSICS[INDEX_TEMPORAL_DEFAULT]]).all()
    assert (output_temperature_space_avg.result[:,0] == [np.mean(INIT_AIR_TEMPERATURE)]).all()
    assert len(observer.temporal_axis) == 1
    assert output_temperature_space_avg.temporal_result[0] == np.mean(INIT_AIR_TEMPERATURE)

def test_gradient_output():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_gradient], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration.
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    assert output_gradient.result[:,0] == approx(np.diff(INIT_AIR_TEMPERATURE) / DX)
    assert output_gradient.temporal_result[0] == approx(T0 / DX)


def test_boundary_output():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_gradient_left], resolution=RESOLUTION, surface=BOX_DEPTH * BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    expected_value = -np.diff(INIT_AIR_TEMPERATURE)[0] / DX
    assert output_gradient_left.size == 1
    assert output_gradient_left.x == approx([0.])
    assert output_gradient_left.result[0,0] == approx(expected_value)
    assert len(observer.temporal_axis) == 1
    assert output_gradient_left.temporal_result[0] == approx(expected_value)

def test_two_outputs():
    # output is computed at ite0
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component1D('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(),
                       [output_temperature_space_avg, output_gradient], resolution=RESOLUTION,
                       surface=BOX_DEPTH*BOX_HEIGHT)
    room.get_grid().set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration
    solver = Solver(component_to_solve_list, DT, TIME_START + DT, observer)
    solver.run()
    solver.visualize()
    assert output_temperature_space_avg.size == 1
    assert (output_temperature_space_avg.x == [X_PHYSICS[INDEX_TEMPORAL_DEFAULT]]).all()
    assert (output_temperature_space_avg.result[:,0] == [np.mean(INIT_AIR_TEMPERATURE)]).all()
    assert len(observer.temporal_axis) == 1
    assert (output_temperature_space_avg.temporal_result == [np.mean(INIT_AIR_TEMPERATURE)]).all()

    assert output_gradient.result[:,0] == approx(np.diff(INIT_AIR_TEMPERATURE) / DX)
    assert output_gradient.temporal_result[0] == approx(T0 / DX)
