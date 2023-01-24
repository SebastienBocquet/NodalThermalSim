from copy import deepcopy
from pathlib import Path
from NodalThermalSim.Component import ConstantComponent, Material, Component1D, Box
from NodalThermalSim.Solver import Solver, Observer, Output
from NodalThermalSim.Physics import FiniteDifferenceTransport
from NodalThermalSim.Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

SOLVER_TYPE = 'implicit'
# factor multiplying the timestep that would be obtained for an explicit time marching
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

# number of nodes discretizing each 1D component
RESOLUTION = 10

THICKNESS_BRICK = 0.1
# thickness of polystyrene layer
THICKNESS_POLY = 0.08

# dimensions of the room
DELTA_X = 2.
DELTA_Y = 2.
DELTA_Z = 2.4

# cell size in 1D components
DX = THICKNESS_BRICK / RESOLUTION

# timestep satisfying stability condition for explicit time marching in air component
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
# timestep satisfying stability condition for explicit time marching in air component
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
# explicit timestep globally satisfying stability condition for explicit time marching
DT = min(DT_AIR, DT_BRICK)

# final timestep for implicit time marching
if SOLVER_TYPE == 'implicit':
    DT *= IMPLICIT_DT_FACTOR

# simulation time in seconds
TIME_START = 0.
TIME_END = 72. * 3600

# location where post processing directory is written
OUTPUT_ROOT_DIR = Path.cwd()

# number of data extractions in time
NB_FRAMES = 15

INIT_WALL_TEMPERATURE = T0
INIT_BOX_TEMPERATURE = T0

# define outputs
output_temperature_brick_left = Output('temperature')
output_heat_flux_brick_left = Output('heat_flux')
output_temperature_brick_right = Output('temperature')
output_heat_flux_brick_right = Output('heat_flux')
output_temperature_poly_left = Output('temperature')
output_heat_flux_poly_left = Output('heat_flux')
output_temperature_poly_right = Output('temperature')
output_heat_flux_poly_right = Output('heat_flux')
output_temperature_box = Output('temperature')
output_heat_flux_left_box = Output('heat_flux', loc='left')
output_heat_flux_right_box = Output('heat_flux', loc='right')

# define components
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE, DELTA_Y * DELTA_Z)
brick_left = Component1D('brick_left', brick, THICKNESS_BRICK, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_brick_left, output_heat_flux_brick_left],
                         resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
poly_left = Component1D('poly_left', poly, THICKNESS_POLY, INIT_WALL_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_poly_left, output_heat_flux_poly_left],
                        resolution = RESOLUTION, surface=DELTA_Y * DELTA_Z)
poly_right = deepcopy(poly_left)
poly_right.material = brick
poly_right.name = 'poly_right'
poly_right.set_outputs([output_temperature_poly_right, output_heat_flux_poly_right])
box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
          INIT_BOX_TEMPERATURE, [output_temperature_box,
                                 output_heat_flux_left_box,
                                 output_heat_flux_right_box])

# define boundary conditions
# use non conservative Dirichlet between wall and room.
# Indeed, conservative Dirichlet requires
bc_diri = BoundaryConditionDirichlet(type='non_conservative')
bc_diri_cons = BoundaryConditionDirichlet(type='conservative')
bc_adia = BoundaryConditionFlux()
FLUX_OUT = -0.
bc_flux_out = BoundaryConditionFlux(type='heat_flux', flux=FLUX_OUT)

HTC = 16.
# boundary conditions must be distinct objects
# because they store physical values as attributes.
bc_htc_box_left = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_box_right = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_poly_left = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_poly_right = BoundaryConditionFlux('htc', htc=HTC)

# y
# ^
# |
#  --> x
#                              adiabatic
#                           --------------
#            |              |            |             |
# EXTERIOR   |  polystyrene |            | polystyrene | flux = -5
# TEMPERATURE|  + brick     |            | (right)     |
#            |  (left)      |            |             |
#                           --------------
#                              adiabatic
#
# ceiling and floor are adiabatic

# associate neighbours and boundary conditions of each component
# box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.
box.get_grid('x').set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})
box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('z').set_boundary({'left': bc_adia, 'right': bc_adia})
box.get_grid('x').set_neighbours({'left': poly_left, 'right': poly_right})

brick_left.get_grid().set_neighbours({'left': air_exterior, 'right': poly_left})
brick_left.get_grid().set_boundary({'left': bc_diri, 'right': bc_diri_cons})

poly_left.get_grid().set_neighbours({'left': brick_left, 'right': box})
poly_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_htc_poly_left})

poly_right.get_grid().set_neighbours({'left': box, 'right': air_interior})
poly_right.get_grid().set_boundary({'left': bc_htc_poly_right, 'right': bc_flux_out})

# list of components to solve
component_to_solve_list = [brick_left, poly_left, box, poly_right]

OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)

solver = Solver(component_to_solve_list, DT, TIME_END, observer, solver_type=SOLVER_TYPE)
# check that components are correctly associated
solver.show_tree()
solver.run()
solver.visualize(OUTPUT_ROOT_DIR / 'htc_bnd_on_walls')

# poly_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})
# poly_right.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_flux_out})
# solver.run()
# solver.visualize(OUTPUT_ROOT_DIR / 'conservative_dirichlet_bnd_on_walls')

#TODO test non conservative dirichlet adjacent to box
