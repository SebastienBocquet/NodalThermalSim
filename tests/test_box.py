import sys
import os
import copy
import numpy as np
import pytest
from pytest import approx
from NodalThermalSim.Component import ConstantComponent, Material, Box, Component1D
from NodalThermalSim.Solver import Solver, Observer, Output
from NodalThermalSim.Physics import FiniteDifferenceTransport
from NodalThermalSim.Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0 - 10.
FLUX = 100.
HTC = 2.

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

output_temperature_wall_left = Output('temperature', RESOLUTION-1)
output_heat_flux_wall_left = Output('heat_flux')
output_heat_flux_wall_left_bc_right = Output('heat_flux', loc='right')
output_temperature_wall_right = Output('temperature', 0)
output_heat_flux_wall_right = Output('heat_flux')
output_heat_flux_wall_right_bc_left = Output('heat_flux', loc='left')
output_temperature_box = Output('temperature', 1)
output_heat_flux_left_box = Output('heat_flux', loc='left')
output_heat_flux_right_box = Output('heat_flux', loc='right')

wall_left = Component1D('wall_left', air, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_wall_left, output_heat_flux_wall_left,
                         output_heat_flux_wall_left_bc_right], resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
wall_right = Component1D('wall_right', air, THICKNESS, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                         [output_temperature_wall_right, output_heat_flux_wall_right,
                          output_heat_flux_wall_right_bc_left], resolution=RESOLUTION, surface=DELTA_Y * DELTA_Z)

bc_diri = BoundaryConditionDirichlet(type='conservative')
bc_adia = BoundaryConditionFlux()


def test_box_bc_value():
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
     box.get_grid('x').set_boundary({'left': BoundaryConditionFlux(flux=-FLUX), 'right': BoundaryConditionFlux(flux=FLUX)})
     box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('x').set_neighbours({'left': air_exterior, 'right': air_interior})
     assert box.get_grid('x').val[1:box.RESOLUTION+1] == approx(INIT_BOX_TEMPERATURE)
     box.update_ghost_node(0., 0)
     # assert box.get_grid('x').val[0] == approx(EXTERIOR_TEMPERATURE)
     assert box.get_grid('x').val[1:3] == approx(INIT_BOX_TEMPERATURE)
     assert box.get_grid('x').get_boundary_gradient('left') * box.material.thermal_conductivity == approx(-FLUX)
     assert box.get_grid('x').get_boundary_gradient('right') * box.material.thermal_conductivity == approx(FLUX)

def test_solver_box_flux():
     # box has opposite flux on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
     box.get_grid('x').set_boundary({'left': BoundaryConditionFlux(flux=FLUX), 'right': BoundaryConditionFlux(flux=FLUX)})
     box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
     component_to_solve_list = [box]
     observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
     solver = Solver(component_to_solve_list, DT, TIME_END, observer)
     solver.run()
     solver.visualize()
     # test that the rate at which temperature increases in the box is equal to the sum of heat flux
     # divided by volume * density * cp
     # the last 1/5 of the simulation is considered.
     increase_rate = (output_temperature_box.temporal_result[-1] - output_temperature_box.temporal_result[-2]) / \
                     (observer.temporal_axis[-1] - observer.temporal_axis[-2])
     expected_increase_rate = 2 * FLUX / (DELTA_X * air.density * air.cp)
     assert increase_rate == approx(expected_increase_rate)

def test_solver_box_htc_constant_component_neighbours():
     # box has opposite flux on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box])
     # box.get_grid('x').set_neighbours({'left': wall_left, 'right': wall_right})
     box.get_grid('x').set_neighbours({'left': air_exterior, 'right': air_exterior})
     box.get_grid('x').set_boundary({'left': BoundaryConditionFlux(type='htc', htc=HTC),
                                     'right': BoundaryConditionFlux(type='htc', htc=HTC)})
     box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
     box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
     component_to_solve_list = [box]
     observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
     solver = Solver(component_to_solve_list, DT, TIME_END, observer)
     solver.run()
     solver.visualize()
     # test that the rate at which temperature increases in the box is equal to the sum of heat flux
     # divided by volume * density * cp
     # the last 1/5 of the simulation is considered.
     increase_rate = (output_temperature_box.temporal_result[-1] - output_temperature_box.temporal_result[-2]) / \
                     (observer.temporal_axis[-1] - observer.temporal_axis[-2])
     delta_temperature_last = EXTERIOR_TEMPERATURE - output_temperature_box.temporal_result[-1]
     delta_temperature_second_to_last = EXTERIOR_TEMPERATURE - output_temperature_box.temporal_result[-2]
     delta_temperature = 0.5 * (delta_temperature_last + delta_temperature_second_to_last)
     expected_increase_rate = 2 * HTC * delta_temperature / (DELTA_X * air.density * air.cp)
     assert increase_rate == approx(expected_increase_rate, rel=0.05)

def test_solver_box_wall_htc_1D_component_neighbours():
     # box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [output_temperature_box, output_heat_flux_left_box,
                                      output_heat_flux_right_box])
     box.get_grid('x').set_neighbours({'left': wall_left, 'right': wall_right})
     wall_left.get_grid().set_neighbours({'left': air_exterior, 'right': box})
     wall_right.get_grid().set_neighbours({'left': box, 'right': air_exterior})
     htc = 8.
     bc_htc = BoundaryConditionFlux(type='htc', htc=htc)
     box.get_grid('x').set_boundary({'left': bc_htc,
                                     'right': bc_htc})
     wall_left.get_grid().set_boundary({'left': bc_diri, 'right': bc_htc})
     wall_right.get_grid().set_boundary({'left': bc_htc, 'right': bc_diri})
     component_to_solve_list = [wall_left, box, wall_right]
     observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
     solver = Solver(component_to_solve_list, DT, TIME_END, observer)
     solver.run()
     solver.visualize()
     # TODO: output time 0 is non physical.
     # test conservativity of heat flux
     OBSERVER_INDEX = 1
     assert output_heat_flux_wall_left_bc_right.temporal_result[OBSERVER_INDEX] ==\
            approx(-output_heat_flux_left_box.temporal_result[OBSERVER_INDEX])
     assert output_heat_flux_right_box.temporal_result[OBSERVER_INDEX] == \
          approx(-output_heat_flux_wall_right_bc_left.temporal_result[OBSERVER_INDEX])

@pytest.mark.xfail(raises=AssertionError)
def test_htc_bnd_not_associated_to_htc_bnd_on_1D_component_neighbour():
     # box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
     box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
               INIT_BOX_TEMPERATURE, [])
     box.get_grid('x').set_neighbours({'left': wall_left, 'right': wall_right})
     wall_left.get_grid().set_neighbours({'left': air_exterior, 'right': box})
     wall_right.get_grid().set_neighbours({'left': box, 'right': air_exterior})
     htc = 8.
     bc_htc = BoundaryConditionFlux(type='htc', htc=htc)
     box.get_grid('x').set_boundary({'left': bc_htc,
                                     'right': bc_htc})
     wall_left.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri})
     wall_right.get_grid().set_boundary({'left': bc_htc, 'right': bc_diri})
     component_to_solve_list = [wall_left, box, wall_right]
     observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)
     solver = Solver(component_to_solve_list, DT, TIME_END, observer)
