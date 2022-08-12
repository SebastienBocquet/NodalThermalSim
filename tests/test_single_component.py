from pytest import approx

import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component2D import ConstantComponent, Material, Component2D
from Solver import Solver, Observer, FiniteDifferenceTransport

EXTERIOR_TEMPERATURE = 298.15
INTERIOR_TEMPERATURE = 293.15
INIT_WALL_TEMPERATURE = 295.

air_exterior = ConstantComponent(EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent(INTERIOR_TEMPERATURE)
CP = 840.0
K = 0.9
DENSITY = 2000.0
brick = Material(CP, DENSITY, K)
RESOLUTION = 10
THICKNESS = 0.14
DX = THICKNESS / RESOLUTION
RESOLUTION = 10
TIME_END = 10 * 3600.
DT = 0.9 * DX**2 / (2 * (K / (DENSITY * CP)))
fd_transport = FiniteDifferenceTransport(DT)
NB_FRAMES = 5
neighbours = {'in': air_interior, 'ext': air_exterior}
wall = Component2D(brick, THICKNESS, INIT_WALL_TEMPERATURE, air_interior, air_exterior, neighbours)
component_list = [wall, air_interior, air_exterior]


def test_constant_component_bc():
    assert air_exterior.get_neighbour_val('in') == approx(EXTERIOR_TEMPERATURE)
    assert air_exterior.get_neighbour_val('ext') == approx(EXTERIOR_TEMPERATURE)

def test_component_bc():
    assert wall.y[:] == approx(INIT_WALL_TEMPERATURE)
    wall.update()
    assert wall.y[0] == approx(INTERIOR_TEMPERATURE)
    assert wall.y[wall.resolution + 1] == approx(EXTERIOR_TEMPERATURE)
    assert wall.y[1] == approx(INIT_WALL_TEMPERATURE)

def test_solver_single_component():
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, fd_transport, DT, TIME_END)
    solver.run()
    # test solution against linear profile between exterior and interior temperature.
    expected_y = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
    assert wall.y == approx(expected_y)

def test_observer():
    # TODO check extreme setup (0 frame, 
    observer = Observer(0, TIME_END / NB_FRAMES, TIME_END, RESOLUTION, DT)
    wall = Component2D(brick, THICKNESS, INIT_WALL_TEMPERATURE, air_interior, air_exterior, neighbours, observer)
    ite_observation0 = (int)(TIME_END / NB_FRAMES / DT)
    assert observer.is_updated(ite_observation0) is True
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, fd_transport, DT, TIME_END)
    solver.run()
    assert observer.update_count == NB_FRAMES


