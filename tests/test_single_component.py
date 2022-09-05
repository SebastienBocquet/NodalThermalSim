import sys
import os
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component
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
DX = THICKNESS / (RESOLUTION - 1)
RESOLUTION = 10
DT = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))

TIME_START = 0.
TIME_END = 20 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature = Output(int(RESOLUTION / 2), var_name='temperature')

neighbours = {'in': air_interior, 'ext': air_exterior}
INIT_WALL_TEMPERATURE = np.ones((RESOLUTION)) * T0

observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature])
wall = Component('wall', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), resolution=RESOLUTION, surface=1., observer=observer)
# wall = Component(brick, THICKNESS, INIT_WALL_TEMPERATURE)
wall.set_neighbours(neighbours)

linear_profile = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
# The temperature is imposed at the ghost nodes. So we add 2 dx to the thickness, to obtain the distance between the ghost nodes.
expected_gradient = (EXTERIOR_TEMPERATURE - INTERIOR_TEMPERATURE) / (THICKNESS + 2 * DX)
wall_linear_profile = Component('wall_linear_profile', brick, THICKNESS, linear_profile[1:RESOLUTION+1], FiniteDifferenceTransport(), resolution=RESOLUTION, surface=1., observer=observer)
wall_linear_profile.set_neighbours(neighbours)

wall_adiabatic = Component('wall_adiabatic', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'in': 'adiabatic', 'ext': 'dirichlet'}, resolution=RESOLUTION, surface=1., observer=observer)
wall_adiabatic.set_neighbours(neighbours)

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
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    # test solution against linear profile between exterior and interior temperature. Ghost nodes are included.
    expected_y = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
    assert wall.y == approx(expected_y)

def test_adiabatic_component():
    wall_adiabatic.update()
    assert wall_adiabatic.y[0] == wall_adiabatic.y[1]
    assert wall_adiabatic.get_boundary_gradient('in') == 0.

