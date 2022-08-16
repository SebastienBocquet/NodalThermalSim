from pytest import approx

import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component, Box
from Solver import Solver, Observer, FiniteDifferenceTransport, FiniteVolume, Output

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0
air_exterior = ConstantComponent(EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent(INTERIOR_TEMPERATURE)
CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0
brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)
CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100
air = Material(CP_AIR, DENSITY_AIR, K_AIR)

RESOLUTION = 10
THICKNESS = 0.14
BOX_SIZE = 2.
BOX_VOLUME = BOX_SIZE ** 3
DX = THICKNESS / (RESOLUTION-1)
RESOLUTION = 10
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT_AIR = 0.9 * BOX_SIZE**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
DT = min(DT_AIR, DT_BRICK)
print('dt air', DT_AIR)
print('dt brick', DT_BRICK)

TIME_START = 0.
TIME_END = 20 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

neighbours = {'in': air_interior, 'ext': air_exterior}
INIT_WALL_TEMPERATURE = np.ones((RESOLUTION)) * T0
wall = Component(brick, THICKNESS, INIT_WALL_TEMPERATURE)
wall.set_neighbours(neighbours)
output = Output(True, int(RESOLUTION / 2))
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, output)
wall_with_observer = Component(brick, THICKNESS, INIT_WALL_TEMPERATURE, observer=observer)
wall_with_observer.set_neighbours(neighbours)

wall_adiabatic = Component(brick, THICKNESS, INIT_WALL_TEMPERATURE, {'in': 'adiabatic', 'ext': 'dirichlet'})
wall_adiabatic.set_neighbours(neighbours)
linear_profile = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
# The temperature is imposed at the ghost nodes. So we add 2 dx to the thickness, to obtain the distance between the ghost nodes.
expected_gradient = (EXTERIOR_TEMPERATURE - INTERIOR_TEMPERATURE) / (THICKNESS + 2 * DX)
wall_linear_profile = Component(brick, THICKNESS, linear_profile[1:RESOLUTION+1])
wall_linear_profile.set_neighbours(neighbours)


def test_constant_component_bc():
    assert air_exterior.get_boundary_value('in') == approx(EXTERIOR_TEMPERATURE)
    assert air_exterior.get_boundary_value('ext') == approx(EXTERIOR_TEMPERATURE)

def test_component_bc_value():
    assert wall.y[1:RESOLUTION+1] == approx(INIT_WALL_TEMPERATURE)
    wall.update()
    assert wall.y[0] == approx(INTERIOR_TEMPERATURE)
    assert wall.y[wall.resolution + 1] == approx(EXTERIOR_TEMPERATURE)
    assert wall.y[1] == approx(INIT_WALL_TEMPERATURE)

def test_component_bc_gradient():
    wall_linear_profile.update()
    assert wall_linear_profile.get_boundary_gradient('in') == approx(-expected_gradient)
    assert wall_linear_profile.get_boundary_gradient('ext') == approx(expected_gradient)

def test_solver_single_component():
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, DT_BRICK, TIME_END)
    solver.run()
    # test solution against linear profile between exterior and interior temperature. Ghost nodes are included.
    expected_y = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
    assert wall.y == approx(expected_y)

def test_observer():
    # TODO check extreme setup. Especially the case of a single frame (typically the last one).
    wall_with_observer.observer.set_frame_ite(DT_BRICK)
    for i in range(NB_FRAMES):
        ite_observation = (int)(i * (int)(OBSERVER_PERIOD / DT_BRICK))
        assert observer.is_updated(ite_observation) is True

    component_to_solve_list = [wall_with_observer]
    solver = Solver(component_to_solve_list, DT_BRICK, TIME_END)
    solver.run()
    assert observer.update_count == NB_FRAMES
    # solver.post()

def test_adiabatic_component():
    wall_adiabatic.update()
    assert wall_adiabatic.y[0] == wall_adiabatic.y[1]
    assert wall_adiabatic.get_boundary_gradient('in') == 0.

def test_box_adiabatic():
    neighbours = {}
    for i in range(6):
        neighbours[f"{i}"] = wall_adiabatic
    wall_adiabatic.update()
    box = Box(air, T0, BOX_VOLUME)
    box.set_neighbours(neighbours)
    component_to_solve_list = [box, wall_adiabatic]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    assert box.y == T0

# def test_box_dirichlet():
#     box = Box(air, T0)
#     box_neighbours = {}
#     for i in range(6):
#         box_neighbours[f"{i}"] = air_exterior
#     box.set_neighbours(box_neighbours)
#     component_to_solve_list = [box]
#     solver = Solver(component_to_solve_list, DT, TIME_END)
#     solver.run()

def test_box_dirichlet():
    wall_linear_profile = Component(brick, THICKNESS, linear_profile[1:RESOLUTION+1])
    print(wall_linear_profile.y)
    output = Output(True, 0)
    observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, output)
    box = Box(air, T0, BOX_VOLUME, observer)
    box_neighbours = {}
    for i in range(6):
        box_neighbours[f"{i}"] = wall_linear_profile
    wall_neighbours = {'in': box, 'ext': air_exterior}
    box.set_neighbours(box_neighbours)
    wall_linear_profile.set_neighbours(wall_neighbours)

    component_to_solve_list = [box, wall_linear_profile]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()
    print(box.observer.ts)
