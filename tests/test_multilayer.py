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
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE)
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
TIME_END = 1 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature = Output(int(RESOLUTION / 2), var_name='temperature', spatial_type='mean')

neighbours = {'in': air_interior, 'ext': air_exterior}
neighbour_faces = {'in': 'ext', 'ext': 'in'}
INIT_WALL_TEMPERATURE = T0

observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature])
wall = Component('wall', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), dx=DX, surface=YZ_SURFACE, observer=observer)
wall.set_neighbours(neighbours, neighbour_faces)

wall_left = copy.deepcopy(wall)
wall_right = copy.deepcopy(wall)

# 50W source
s = Source(50. / ROOM_VOLUME)
air = Room('air', air, ROOM_WIDTH, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(), dx=DX, surface=YZ_SURFACE, source=s, observer=observer)

wall_left.set_neighbours({'in': air_exterior, 'ext': air}, {'in': 'ext', 'ext': 'left'})
wall_left.name = 'wall left'
wall_right.set_neighbours({'in': air, 'ext': air_exterior}, {'in': 'right', 'ext': 'in'})
wall_right.name = 'wall right'

neighbours = {'left': wall_left, 'right': wall_right}
neighbour_faces = {'left': 'ext', 'right': 'in'}
air.set_neighbours(neighbours, neighbour_faces)


def test_wall_air_wall():
    component_to_solve_list = [wall_left, air, wall_right]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()

def test_show_tree():
    air.show_tree()

