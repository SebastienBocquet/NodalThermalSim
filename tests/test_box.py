import sys
import os
import copy
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Box, Component1D
from Solver import Solver, Observer, Output
from Physics import FiniteDifferenceTransport
from Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0 - 10.
FLUX = -1000. # negative outward flux, so positive inward flux.

DELTA_X = 0.1
DELTA_Y = 0.1
DELTA_Z = 0.1

# these two components have normal along X
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0
brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)
CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100
air = Material(CP_AIR, DENSITY_AIR, K_AIR)

INIT_WALL_TEMPERATURE = T0
INIT_BOX_TEMPERATURE = T0

RESOLUTION = 10
THICKNESS = 0.1

DX = THICKNESS / (RESOLUTION - 1)
RESOLUTION = 10
# DT = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))

TIME_START = 0.
TIME_END = 120. #0.5 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature_wall_left = Output('temperature', int(RESOLUTION / 2))
output_temperature_wall_right = Output('temperature', int(RESOLUTION / 2))
output_temperature_box = Output('temperature', 1)

wall_left = Component1D('wall_left', air, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_wall_left], resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
wall_right = Component1D('wall_right', air, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                         [output_temperature_wall_right], resolution=RESOLUTION, surface=DELTA_Y * DELTA_Z)

bc_diri = BoundaryConditionDirichlet()
bc_adia = BoundaryConditionFlux()
bc_flux_left = BoundaryConditionFlux(flux=FLUX)
bc_flux_right = BoundaryConditionFlux(flux=-FLUX)


def test_box_bc_value():
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
     box.get_grid('x').set_boundary({'left': bc_diri, 'right': BoundaryConditionFlux(flux=FLUX)})
     box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('x').set_neighbours({'left': air_exterior, 'right': air_interior})
     assert box.get_grid('x').val[1:box.RESOLUTION+1] == approx(INIT_BOX_TEMPERATURE)
     box.update_ghost_node(0., 0)
     assert box.get_grid('x').val[0] == approx(EXTERIOR_TEMPERATURE)
     assert box.get_grid('x').val[1:3] == approx(INIT_BOX_TEMPERATURE)
     assert box.get_grid('x').get_boundary_gradient('right') * box.material.thermal_conductivity == approx(-FLUX)

def test_solver_box_flux():
     # box has opposite flux on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
     box.get_grid('x').set_boundary({'left': BoundaryConditionFlux(flux=FLUX), 'right': BoundaryConditionFlux(flux=-FLUX)})
     box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
     component_to_solve_list = [box]
     observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
     solver = Solver(component_to_solve_list, DT, TIME_END, observer)
     solver.run()
     solver.visualize()
     # test solution against linear profile between exterior and interior temperature. Ghost nodes are included.
     assert box.get_grid('x').val[2] == approx(INIT_BOX_TEMPERATURE)

def test_solver_box_wall():
     # box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
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
     solver.visualize()
     # test solution against linear profile between exterior and interior temperature. Ghost nodes are included.
     assert box.get_grid('x').val[box.get_grid().FIRST_PHYS_VAL_INDEX['left']] == \
            approx(0.5 * (INTERIOR_TEMPERATURE + EXTERIOR_TEMPERATURE))
