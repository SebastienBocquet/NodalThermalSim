import sys
import os
import copy
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component, Room, Source
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport, FiniteVolume

T0 = 273.15 + 25.
INIT_WALL_TEMPERATURE = T0
EXTERIOR_TEMPERATURE = 273.15 + 33.
FLUX = -200.
INTERIOR_TEMPERATURE = T0
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
ROOM_WIDTH = 2.
ROOM_HEIGHT = 2.5
ROOM_DEPTH = 2.
ROOM_VOLUME = ROOM_WIDTH * ROOM_HEIGHT * ROOM_DEPTH
YZ_SURFACE = ROOM_HEIGHT * ROOM_DEPTH
DX = THICKNESS / (RESOLUTION - 1)
RESOLUTION = 10
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
DT = min(DT_AIR, DT_BRICK)
print('dt air', DT_AIR)
print('dt brick', DT_BRICK)

TIME_START = 0.
TIME_END = 24 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature = Output('temperature', spatial_type='mean')
output_heat_flux = Output(var_name='heat_flux', spatial_type='raw')

neighbour_faces = {'left': 'right', 'right': 'left'}

observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature, output_heat_flux])

# air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE)
# air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE)
# air_exterior = Component('air', air, THICKNESS, EXTERIOR_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'left': 'adiabatic', 'right': 'dirichlet'}, dx=DX, surface=YZ_SURFACE, observer=observer_)
# observer_ = copy.deepcopy(observer)
# air_interior = Component('air', air, THICKNESS, INTERIOR_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'left': 'dirichlet', 'right': 'adiabatic'}, dx=DX, surface=YZ_SURFACE, observer=observer_)

observer_ = copy.deepcopy(observer)
wall_left = Component('wall_left', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'left': 'flux', 'right': 'dirichlet'}, dx=DX, surface=YZ_SURFACE, observer=observer_, flux={'left': FLUX, 'right': None})
observer_ = copy.deepcopy(observer)
wall_right = Component('wall_right', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), boundary_type={'left': 'dirichlet', 'right': 'flux'}, dx=DX, surface=YZ_SURFACE, observer=observer_, flux={'right': None, 'right': -FLUX})

# 50W source
# s = Source(50. / ROOM_VOLUME)
observer_ = copy.deepcopy(observer)
air = Component('air', air, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), dx=DX, surface=YZ_SURFACE, observer=observer_)

# air_exterior.set_neighbours({'left': None, 'right': wall_left}, {'left': 'right', 'right': 'left'})
# air_interior.set_neighbours({'left': wall_right, 'right': None}, {'left': 'right', 'right': 'left'})
wall_left.set_neighbours({'left': None, 'right': air}, {'left': 'right', 'right': 'left'})
wall_right.set_neighbours({'left': air, 'right': None}, {'left': 'right', 'right': 'left'})
air.set_neighbours({'left': wall_left, 'right': wall_right})

# TODO: use only two components. Check conservation of heat flux.
# check that the 'electric resistance' is respected.
def test_wall_air_wall():
    component_to_solve_list = [wall_left, air, wall_right]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()

def test_show_tree():
    component_to_solve_list = [wall_left, air, wall_right]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.show_tree()

