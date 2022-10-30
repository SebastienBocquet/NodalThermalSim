import sys
import os
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport

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

INIT_AIR_TEMPERATURE = T0 * np.linspace(0, RESOLUTION, num=RESOLUTION)

neighbours = {'left': air_interior, 'right': air_exterior}

output_temperature = Output('temperature', int(RESOLUTION / 2))
output_temperature_space_avg = Output(var_name='temperature', spatial_type='mean')
output_gradient_ext = Output('temperature_gradient', loc='left')
output_gradient_in = Output('temperature_gradient', loc='right')

def test_observer():
    observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature], resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT)
    room.set_neighbours(neighbours)

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
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.set_observer(observer)
    solver.run()
    assert observer.update_count == NB_FRAMES

def test_observer_ite0():
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    # add a set_observer function to Component
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature], resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT)
    room.set_neighbours(neighbours)

    # TODO check extreme setup. Especially the case of a single frame (typically the last one).
    expected_observed_ite = 0
    observed_ite = [expected_observed_ite]
    print(observer.ite_extraction)
    assert (observed_ite == observer.ite_extraction).all()
    assert observer.is_updated(expected_observed_ite) is True

    component_to_solve_list = [room]
    solver = Solver(component_to_solve_list, DT, TIME_START + DT)
    solver.set_observer(observer)
    solver.run()
    assert observer.update_count == 1

def test_raw_output():
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature], resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT)
    room.set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration.
    solver = Solver(component_to_solve_list, DT, TIME_START + DT)
    solver.set_observer(observer)
    solver.run()
    solver.compute_post()
    assert output_temperature.size == RESOLUTION
    assert (output_temperature.x == np.linspace(0, RESOLUTION * DX, num=RESOLUTION)).all()
    assert (output_temperature.result[:,0] == INIT_AIR_TEMPERATURE).all()
    assert len(observer.temporal_axis) == 1
    assert observer.temporal_axis[:] == [TIME_START]
    assert (room.temporal_output[:,0] == [INIT_AIR_TEMPERATURE[(int)(0.5 * RESOLUTION)]]).all()

def test_spatial_avg_output():
    observer = Observer(TIME_START, DT, TIME_START + DT, DT)
    # observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature_space_avg])
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), [output_temperature_space_avg], resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT)
    room.set_neighbours(neighbours)
    component_to_solve_list = [room]
    # run one iteration
    solver = Solver(component_to_solve_list, DT, TIME_START + DT)
    solver.set_observer(observer)
    solver.run()
    solver.compute_post()
    x = np.linspace(0, RESOLUTION * DX, num=RESOLUTION)
    assert output_temperature_space_avg.size == 1
    assert (output_temperature_space_avg.x == [x[(int)(0.5 * RESOLUTION)]]).all()
    assert (output_temperature_space_avg.result[:,0] == [np.mean(INIT_AIR_TEMPERATURE)]).all()
    assert len(observer.temporal_axis) == 1
    assert (room.temporal_output[:,0] == [np.mean(INIT_AIR_TEMPERATURE)]).all()

# TODO: test other outputs