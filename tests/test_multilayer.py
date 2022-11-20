import sys
import os
import copy
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, BoundaryCondition, Component
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
K_AIR_2 = 2 * K_AIR
K_AIR_3 = 3 * K_AIR
air1 = Material(CP_AIR, DENSITY_AIR, K_AIR)
air2 = Material(CP_AIR, DENSITY_AIR, K_AIR_2)
air3 = Material(CP_AIR, DENSITY_AIR, K_AIR_3)

RESOLUTION = 10
THICKNESS = 0.14
ROOM_WIDTH = 2.
ROOM_HEIGHT = 2.5
ROOM_DEPTH = 2.
ROOM_VOLUME = ROOM_WIDTH * ROOM_HEIGHT * ROOM_DEPTH
YZ_SURFACE = ROOM_HEIGHT * ROOM_DEPTH
DX = THICKNESS / RESOLUTION
RESOLUTION = 10
DT = 0.9 * DX**2 / (2 * (K_AIR_3 / (DENSITY_AIR * CP_AIR)))

TIME_START = 0.
TIME_END = 120.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

output_temperature = Output('temperature', spatial_type='raw')
output_heat_flux = Output('heat_flux', spatial_type='raw')

neighbour_faces = {'left': 'right', 'right': 'left'}

observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)

bc_diri = BoundaryCondition('dirichlet')
bc_adia = BoundaryCondition('adiabatic')
bc_flux_left = BoundaryCondition('flux', FLUX)
bc_flux_right = BoundaryCondition('flux', -FLUX)
wall_left = Component('wall_left', air1, THICKNESS, INIT_WALL_TEMPERATURE,
                      FiniteDifferenceTransport(),
                      [output_temperature, output_heat_flux],
                      dx=DX, surface=YZ_SURFACE)
wall_middle = Component('wall_middle', air2, THICKNESS, INIT_WALL_TEMPERATURE,
                FiniteDifferenceTransport(),
                [copy.deepcopy(output_temperature), copy.deepcopy(output_heat_flux)],
                dx=DX, surface=YZ_SURFACE)
wall_right = Component('wall_right', air3, THICKNESS, INIT_WALL_TEMPERATURE,
                       FiniteDifferenceTransport(),
                       [copy.deepcopy(output_temperature), copy.deepcopy(output_heat_flux)],
                       dx=DX, surface=YZ_SURFACE)

wall_left.set_neighbours({'left': None, 'right': wall_middle})
wall_left.set_boundary({'left': bc_flux_left, 'right': bc_diri})

wall_right.set_neighbours({'left': wall_middle, 'right': None})
wall_right.set_boundary({'left': bc_diri, 'right': bc_flux_right})

wall_middle.set_neighbours({'left': wall_left, 'right': wall_right})

def test_wall_air_wall():
    # check that the 'electric resistance' analogy is respected.
    component_to_solve_list = [wall_left, wall_middle, wall_right]
    solver = Solver(component_to_solve_list, DT, TIME_END, observer)
    solver.run()
    solver.visualize()
    expected_flux = np.ones((RESOLUTION)) * FLUX
    thermal_conductivities = [K_AIR, K_AIR_2, K_AIR_3]
    for i in range(len(component_to_solve_list)):
        # check conservation of heat flux through all layers
        assert np.allclose(component_to_solve_list[i].outputs[1].result[:,NB_FRAMES-1], expected_flux)
        # check that temperature profile is linear
        # and its slope corresponds to the thermal resistance analogy
        resistance = THICKNESS / (YZ_SURFACE * thermal_conductivities[i])
        expected_grad_temp = np.ones((RESOLUTION)) * resistance * (FLUX * YZ_SURFACE) / THICKNESS
        grad_temp = np.diff(component_to_solve_list[i].outputs[0].result[:, NB_FRAMES - 1]) / DX
        assert np.allclose(grad_temp, expected_grad_temp)

def test_show_tree():
    component_to_solve_list = [wall_left, wall_middle, wall_right]
    solver = Solver(component_to_solve_list, DT, TIME_END, observer)
    solver.show_tree()
