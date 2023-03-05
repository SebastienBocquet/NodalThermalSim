from copy import deepcopy
from pathlib import Path
from NodalThermalSim.Component import ConstantComponent, Material, Component1D, Box
from NodalThermalSim.Solver import Solver, Observer, Output
from NodalThermalSim.Physics import FiniteDifferenceTransport
from NodalThermalSim.Grid import BoundaryConditionDirichlet, BoundaryConditionFlux

SOLVER_TYPE = 'implicit'
# factor multiplying the timestep that would be obtained for an explicit time marching
IMPLICIT_DT_FACTOR = 40

T0 = 273.15
T_AMB = T0 + 19.
T_GROUND = T0 + 13
EXTERIOR_TEMPERATURE = T0 + 2.
INTERIOR_TEMPERATURE = T_AMB

CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 10

CP_BRICK = 840.0
K_BRICK = 0.9
DENSITY_BRICK = 2000.0

CP_POLYSTYRENE = 1300.
DENSITY_POLYSTYRENE = 20.
K_POLYSTYRENE = 0.035

CP_OUATE = 1300.
DENSITY_OUATE = 35.
K_OUATE = 0.05

air = Material(CP_AIR, DENSITY_AIR, K_AIR)
brick = Material(CP_BRICK, DENSITY_BRICK, K_BRICK)
poly = Material(CP_POLYSTYRENE, DENSITY_POLYSTYRENE, K_POLYSTYRENE)
ouate = Material(CP_OUATE, DENSITY_OUATE, K_OUATE)

THICKNESS_BRICK = 0.1
# thickness of polystyrene layer
THICKNESS_POLY = 0.08
THICKNESS_GROUND = 0.3
THICKNESS_OUATE = 0.33
THICKNESS_STUFF = 0.3

# dimensions of the room
DELTA_X = 3.
DELTA_Y = 3.
DELTA_Z = 2.4

# cell size in 1D components
DX = 0.01

# timestep satisfying stability condition for explicit time marching in air component
DT_AIR = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))
# timestep satisfying stability condition for explicit time marching in air component
DT_BRICK = 0.9 * DX**2 / (2 * (K_BRICK / (DENSITY_BRICK * CP_BRICK)))
DT_POLY = 0.9 * DX**2 / (2 * (K_POLYSTYRENE / (DENSITY_POLYSTYRENE * CP_POLYSTYRENE)))
DT_OUATE = 0.9 * DX**2 / (2 * (K_OUATE / (DENSITY_OUATE * CP_OUATE)))
# explicit timestep globally satisfying stability condition for explicit time marching
DT = min(DT_AIR, DT_BRICK, DT_POLY, DT_OUATE)

# final timestep for implicit time marching
if SOLVER_TYPE == 'implicit':
    DT *= IMPLICIT_DT_FACTOR

# simulation time in seconds
TIME_START = 0.
TIME_END = 96. * 3600

# location where post processing directory is written
OUTPUT_ROOT_DIR = Path.cwd()

# number of data extractions in time
NB_FRAMES = 15

INIT_AMBIANT_TEMPERATURE = T_AMB
INIT_BOX_TEMPERATURE = T_AMB
INIT_BRICK_TEMPERATURE = 0.5 * EXTERIOR_TEMPERATURE + 0.5 * INTERIOR_TEMPERATURE
INIT_FLOOR_TEMPERATURE = 0.25 * T_GROUND + 0.75 * T_AMB

# define outputs
# output_temperature_brick_left = Output('temperature')
# output_heat_flux_brick_left = Output('heat_flux')
# output_temperature_poly_left = Output('temperature')
# output_heat_flux_poly_left = Output('heat_flux')
output_temperature_box = Output('temperature')
output_heat_flux_left_box = Output('heat_flux', loc='left')
output_heat_flux_right_box = Output('heat_flux', loc='right')
output_temperature_ground = Output('temperature')
output_heat_flux_ground = Output('heat_flux')
output_temperature_top = Output('temperature')
output_temperature_stuff = Output('temperature')

# define components
air_exterior = ConstantComponent('air_exterior', EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent('air_interior', INTERIOR_TEMPERATURE)
air_roof = ConstantComponent('air_roof', 0.75 * EXTERIOR_TEMPERATURE + 0.25 * INTERIOR_TEMPERATURE)
ground = ConstantComponent('ground', T_GROUND)

brick_right = Component1D('brick_right', brick, THICKNESS_BRICK, INIT_BRICK_TEMPERATURE,
                          FiniteDifferenceTransport(),
                          [],
                          dx=DX, surface=DELTA_Y * DELTA_Z)
brick_north = Component1D('brick_north', brick, THICKNESS_BRICK, INIT_BRICK_TEMPERATURE, FiniteDifferenceTransport(),
                          [],
                         dx=DX, surface=DELTA_X * DELTA_Z)
brick_ground = Component1D('brick_ground', brick, THICKNESS_GROUND, INIT_FLOOR_TEMPERATURE, FiniteDifferenceTransport(),
                          [output_temperature_ground, output_heat_flux_ground],
                           dx=DX, surface=DELTA_X * DELTA_Y)

poly_right = Component1D('poly_right', poly, THICKNESS_POLY, INIT_AMBIANT_TEMPERATURE, FiniteDifferenceTransport(),
                         [],
                         dx=DX, surface=DELTA_Y * DELTA_Z)
poly_north = Component1D('poly_north', poly, THICKNESS_POLY, INIT_AMBIANT_TEMPERATURE, FiniteDifferenceTransport(),
                        [],
                        dx=DX, surface=DELTA_X * DELTA_Z)
ouate_top = Component1D('ouate_top', ouate, THICKNESS_OUATE, INIT_AMBIANT_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_top],
                        dx=DX, surface=DELTA_X * DELTA_Y)
internal_stuff = Component1D('internal_stuff', ouate, THICKNESS_STUFF, INIT_AMBIANT_TEMPERATURE, FiniteDifferenceTransport(),
                        [output_temperature_stuff],
                        dx=DX, surface=DELTA_Y * DELTA_Z)

box = Box('box', air, DELTA_X, DELTA_Y, DELTA_Z,
          INIT_BOX_TEMPERATURE, [output_temperature_box,
                                 output_heat_flux_left_box,
                                 output_heat_flux_right_box])

# define boundary conditions
# use non conservative Dirichlet between wall and room.
# Indeed, conservative Dirichlet requires
bc_diri_cons = BoundaryConditionDirichlet(type='conservative')
bc_adia = BoundaryConditionFlux()
FLUX_OUT = -0.
bc_flux_out = BoundaryConditionFlux(type='heat_flux', flux=FLUX_OUT)

HTC = 1.
# boundary conditions must be distinct objects
# because they store physical values as attributes.
bc_htc_box_left = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_box_right = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_box_north = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_box_ground = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_box_top = BoundaryConditionFlux('htc', htc=HTC)

bc_htc_stuff = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_poly_right = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_poly_north = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_brick_ground = BoundaryConditionFlux('htc', htc=HTC)
bc_htc_poly_top = BoundaryConditionFlux('htc', htc=HTC)

# y
# ^
# |
#  --> x
#                 adiabatic
#              --------------
#              |            |                 |
#  adiabatic   |            | polystyrene     | air exterior
#              |            | + brick (right) |
#              |            |                 |
#              --------------
#                 adiabatic
#
# floor is connected to brick_ground
# ceiling is connected to ouate, and ouate is connecte to air_roof

# associate neighbours and boundary conditions of each component
# box is connected to walls on the x-left and x-right faces, and adiabatic on other faces.

box.get_grid('x').set_boundary({'left': bc_htc_box_left, 'right': bc_htc_box_right})
# box.get_grid('x').set_boundary({'left': bc_flux_box_left, 'right': bc_flux_box_right})
# box.get_grid('x').set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})
box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_adia})
# box.get_grid('y').set_boundary({'left': bc_adia, 'right': bc_htc_box_north})
box.get_grid('z').set_boundary({'left': bc_htc_box_ground, 'right': bc_htc_box_top})

box.get_grid('x').set_neighbours({'left': internal_stuff, 'right': poly_right})
box.get_grid('y').set_neighbours({'left': None, 'right': None})
box.get_grid('z').set_neighbours({'left': brick_ground, 'right': ouate_top})

# brick_left.get_grid().set_neighbours({'left': air_exterior, 'right': poly_left})
# brick_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

brick_right.get_grid().set_neighbours({'left': poly_right, 'right': air_exterior})
brick_right.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

# brick_north.get_grid().set_neighbours({'left': poly_north, 'right': air_exterior})
# brick_north.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

brick_ground.get_grid().set_neighbours({'left': ground, 'right': box})
brick_ground.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_htc_brick_ground})

# poly_left.get_grid().set_neighbours({'left': brick_left, 'right': box})
# poly_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_htc_poly_left})
# poly_left.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

poly_right.get_grid().set_neighbours({'left': box, 'right': brick_right})
poly_right.get_grid().set_boundary({'left': bc_htc_poly_right, 'right': bc_diri_cons})
# poly_right.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

# poly_north.get_grid().set_neighbours({'left': box, 'right': brick_north})
# poly_north.get_grid().set_boundary({'left': bc_htc_poly_north, 'right': bc_diri_cons})
# poly_north.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_diri_cons})

ouate_top.get_grid().set_neighbours({'left': box, 'right': air_roof})
ouate_top.get_grid().set_boundary({'left': bc_htc_poly_top, 'right': bc_diri_cons})

internal_stuff.get_grid().set_neighbours({'left': air_interior, 'right': box})
internal_stuff.get_grid().set_boundary({'left': bc_diri_cons, 'right': bc_htc_stuff})

# list of components to solve
component_to_solve_list = [box, poly_right, brick_right, ouate_top, brick_ground, internal_stuff]
# component_to_solve_list = [box]

OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)
observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, DT)

solver = Solver(component_to_solve_list, DT, TIME_END, observer, solver_type=SOLVER_TYPE)
solver.show_tree()
solver.run()
solver.visualize(OUTPUT_ROOT_DIR / 'htc_bnd_on_walls')

#TODO simulate my house with a single box. The house temperature evolves from 19deg to 135deg in 30h,
# for an external temperature of 0deg.
