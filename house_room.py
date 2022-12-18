import sys
import os
from copy import deepcopy
import logging
from pathlib import Path
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component1D, Box
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport
from Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

SOLVER_TYPE = 'implicit'
IMPLICIT_DT_FACTOR = 40

T0 = 273.15 + 19.
INIT_WALL_TEMPERATURE = T0
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0

CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100

CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0

CP_POLYSTYRENE = 1300.
DENSITY_POLYSTYRENE = 20.
K_POLYSTYRENE = 0.035

air = Material(CP_AIR, DENSITY_AIR, K_AIR)
brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)
poly = Material(CP_POLYSTYRENE, DENSITY_POLYSTYRENE, K_POLYSTYRENE)

RESOLUTION = 10
THICKNESS_BRICK = 0.1
THICKNESS_POLY = 0.08
DELTA_X = 2.
DELTA_Y = 2.
DELTA_Z = 2.4
DX = THICKNESS_BRICK / RESOLUTION
RESOLUTION = 10
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT = min(DT_AIR, DT_BRICK)

if SOLVER_TYPE == 'implicit':
    DT *= IMPLICIT_DT_FACTOR

TIME_START = 0.
TIME_END = 12. * 3600

OUTPUT_ROOT_DIR = Path('.')
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

INIT_WALL_TEMPERATURE = T0
INIT_BOX_TEMPERATURE = T0

NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature_brick_left = Output('temperature', int(RESOLUTION / 2))
output_temperature_brick_right = Output('temperature', int(RESOLUTION / 2))
output_temperature_poly_left = Output('temperature', int(RESOLUTION / 2))
output_temperature_poly_right = Output('temperature', int(RESOLUTION / 2))
output_temperature_box = Output('temperature', 1)

air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
brick_left = Component1D('brick_left', brick, THICKNESS_BRICK, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_brick_left], resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
# brick_right = deepcopy(brick_left)
# brick_right.name = 'brick_right'
# brick_right.set_outputs([output_temperature_brick_right])
poly_left = Component1D('poly_left', poly, THICKNESS_POLY, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_poly_left], resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
poly_right = deepcopy(poly_left)
poly_right.material = brick
poly_right.name = 'poly_right'
poly_right.set_outputs([output_temperature_poly_right])
box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
          INIT_BOX_TEMPERATURE, [output_temperature_box])

bc_diri = BoundaryConditionDirichlet(type='non_conservative')
bc_diri_cons = BoundaryConditionDirichlet(type='conservative')
bc_adia = BoundaryConditionFlux()

# box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('x').set_neighbours({'left': poly_left, 'right': poly_right})

brick_left.get_grid().set_neighbours({'left': air_exterior, 'right': poly_left})
# brick_right.get_grid().set_neighbours({'left': poly_right, 'right': air_interior})
brick_left.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri_cons})
# brick_right.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri})

poly_left.get_grid().set_neighbours({'left': brick_left, 'right': box})
poly_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri})
poly_right.get_grid().set_neighbours({'left': box, 'right': air_interior})
poly_right.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri})

component_to_solve_list = [brick_left, poly_left, box, poly_right]
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
solver = Solver(component_to_solve_list, DT, TIME_END, observer, solver_type=SOLVER_TYPE)
solver.show_tree()
solver.run()
solver.visualize(OUTPUT_ROOT_DIR)