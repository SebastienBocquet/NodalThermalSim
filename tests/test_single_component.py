import sys
import os
import copy
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component, Room
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport, FiniteVolume

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE)
CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0
brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)

RESOLUTION = 10
THICKNESS = 0.14
DX = THICKNESS / (RESOLUTION - 1)
RESOLUTION = 10
DT = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))

TIME_START = 0.
TIME_END = 48 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature = Output('temperature', int(RESOLUTION / 2))

neighbours = {'left': air_interior, 'right': air_exterior}
neighbour_faces = {'left': 'right', 'right': 'left'}
INIT_WALL_TEMPERATURE = T0

observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature])
wall = Component('wall', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), dx=DX, surface=1., observer=observer)
wall.set_neighbours(neighbours, neighbour_faces)

linear_profile = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
# The temperature is imposed at the ghost nodes. So we add 2 dx to the thickness, to obtain the distance between the ghost nodes.
expected_gradient = (EXTERIOR_TEMPERATURE - INTERIOR_TEMPERATURE) / (THICKNESS + 2 * DX)

wall_adiabatic = Component('wall_adiabatic', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'left': 'adiabatic', 'right': 'dirichlet'}, resolution=RESOLUTION, surface=1., observer=observer)
wall_adiabatic.set_neighbours(neighbours, neighbour_faces)


def test_constant_component_bc():
    assert air_exterior.get_boundary_value('left') == approx(EXTERIOR_TEMPERATURE)
    assert air_exterior.get_boundary_value('right') == approx(EXTERIOR_TEMPERATURE)

def test_component_bc_value():
    assert wall.y[1:RESOLUTION+1] == approx(INIT_WALL_TEMPERATURE)
    wall.update(0.)
    assert wall.y[0] == approx(INTERIOR_TEMPERATURE)
    assert wall.y[wall.resolution + 1] == approx(EXTERIOR_TEMPERATURE)
    assert wall.y[1] == approx(INIT_WALL_TEMPERATURE)

def test_solver_single_component():
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()
    # test solution against linear profile between exterior and interior temperature. Ghost nodes are included.
    expected_y = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
    assert wall.y == approx(expected_y)
    assert wall.get_boundary_gradient('left') == approx(-expected_gradient)
    assert wall.get_boundary_gradient('right') == approx(expected_gradient)

def test_adiabatic_component():
    wall_adiabatic.update(0.)
    assert wall_adiabatic.y[0] == wall_adiabatic.y[1]
    assert wall_adiabatic.get_boundary_gradient('left') == 0.


