import sys
import os
import copy
import logging
from pathlib import Path
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component1D, Box
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport
from Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

T0 = 273.15 + 19.
INIT_WALL_TEMPERATURE = T0
EXTERIOR_TEMPERATURE = 273.15 + 33.
FLUX = -200.
INTERIOR_TEMPERATURE = T0

CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100

CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0

air = Material(CP_AIR, DENSITY_AIR, K_AIR)

brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)

RESOLUTION = 10
THICKNESS = 0.07
DELTA_X = 2.
DELTA_Y = 2.
DELTA_Z = 2.4
DX = THICKNESS / RESOLUTION
RESOLUTION = 10
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT = min(DT_AIR, DT_BRICK)

TIME_START = 0.
TIME_END = 12. * 3600

OUTPUT_ROOT_DIR = Path('.')
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

INIT_WALL_TEMPERATURE = T0
INIT_BOX_TEMPERATURE = T0

NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature_wall_left = Output('temperature', int(RESOLUTION / 2))
output_temperature_wall_right = Output('temperature', int(RESOLUTION / 2))
output_temperature_box = Output('temperature', 1)

air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
wall_left = Component1D('wall_left', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_wall_left], resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
wall_right = Component1D('wall_right', brick, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                         [output_temperature_wall_right], resolution=RESOLUTION, surface=DELTA_Y * DELTA_Z)
box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
          INIT_BOX_TEMPERATURE, [output_temperature_box])

bc_diri = BoundaryConditionDirichlet(type='non_conservative')
bc_adia = BoundaryConditionFlux()

# box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('x').set_neighbours({'left': wall_left, 'right': wall_right})
wall_left.get_grid().set_neighbours({'left': air_exterior, 'right': box})
wall_right.get_grid().set_neighbours({'left': box, 'right': air_interior})
wall_left.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri})
wall_right.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri})
component_to_solve_list = [wall_left, box, wall_right]
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
solver = Solver(component_to_solve_list, DT, TIME_END, observer)
solver.run()
solver.visualize(OUTPUT_ROOT_DIR)