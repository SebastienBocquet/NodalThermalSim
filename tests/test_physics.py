import sys
import os
import copy
import numpy as np
from pytest import approx
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

THICKNESS = 0.14
BOX_WIDTH = 2.
BOX_HEIGHT = 2.5
BOX_VOLUME = BOX_WIDTH * BOX_HEIGHT * 1.
WALL_RESOLUTION = 10
DX = THICKNESS / (WALL_RESOLUTION-1)
WALL_SURFACE = 2.5 * 1.
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))

BOX_RESOLUTION = (int)(BOX_WIDTH / DX) + 1
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
DT = min(DT_AIR, DT_BRICK)
print('dt air', DT_AIR)
print('dt brick', DT_BRICK)

TIME_START = 0.
TIME_END = 12 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

INIT_WALL_TEMPERATURE = np.ones((WALL_RESOLUTION)) * T0
INIT_AIR_TEMPERATURE = np.ones((BOX_RESOLUTION)) * T0
output_temperature = Output(int(WALL_RESOLUTION / 2))
output_gradient_ext = Output(0, var_name='temperature_gradient', loc='ext')
output_gradient_in = Output(0, var_name='temperature_gradient', loc='in')
# TODO set output after init. Create a single observer object (behaviour class). 
# Solver, Observer should also be single object.
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature, output_gradient_ext])
wall1 = Component('wall1', brick, THICKNESS, INIT_WALL_TEMPERATURE, resolution=WALL_RESOLUTION, surface=WALL_SURFACE, observer=observer)
wall2 = copy.deepcopy(wall1)
wall2.name = 'wall2'
wall2.observer.outputs = [output_temperature, output_gradient_in]
# TODO use DX as input to Component
# allow initialization by a float
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature, output_gradient_in, output_gradient_ext])
air_component = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, resolution=BOX_RESOLUTION, surface=WALL_SURFACE, observer=observer)
neighbours = {'in': air_exterior, 'ext': air_component}
wall1.set_neighbours(neighbours)
neighbours = {'in': air_component, 'ext': air_exterior}
wall2.set_neighbours(neighbours)
neighbours = {'in': wall1, 'ext': wall2}
air_component.set_neighbours(neighbours)

def test_solver_brick_and_air():
    component_to_solve_list = [wall1, air_component, wall2]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()
