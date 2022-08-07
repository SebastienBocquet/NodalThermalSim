from pytest import approx

import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component2D import Air, Component2D, Solver, Observer, FiniteDifferenceTransport

EXTERIOR_TEMPERATURE = 298.15
INTERIOR_TEMPERATURE = 293.15
INIT_WALL_TEMPERATURE = 295.

air_exterior = Air("air_exterior", EXTERIOR_TEMPERATURE)
air_interior = Air("air_interior", INTERIOR_TEMPERATURE)
CP = 840.
K = 0.9
DENSITY = 2000.
RESOLUTION = 10
THICKNESS = 0.14
DX = THICKNESS / RESOLUTION
RESOLUTION = 10
TIME_END = 10 * 3600.
DT = 0.9 * DX**2 / (2 * (K / (DENSITY * CP)))
fd_transport = FiniteDifferenceTransport(DT)
NB_FRAMES = 5
wall = Component2D("wall", CP, DENSITY, K, THICKNESS, INIT_WALL_TEMPERATURE, air_interior, air_exterior)
component_list = [wall, air_interior, air_exterior]

def test_single_component_bc():
    assert air_exterior.get_bc_val_ext() == approx(EXTERIOR_TEMPERATURE)
    assert wall.y[:] == approx(INIT_WALL_TEMPERATURE)

    wall.update_bc()
    assert wall.y[0] == approx(INTERIOR_TEMPERATURE)
    assert wall.y[wall.resolution + 1] == approx(EXTERIOR_TEMPERATURE)
    assert wall.y[1] == approx(INIT_WALL_TEMPERATURE)

def test_solver_single_component():
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, fd_transport, DT, TIME_END)
    solver.run()
    print("test assertion solver")
    expected_y = np.linspace(INTERIOR_TEMPERATURE, EXTERIOR_TEMPERATURE, RESOLUTION+2)
    assert wall.y == approx(expected_y)
    # TODO check result versus linear solution profile.

def test_observer():
    # TODO check extreme setup (0 frame, 
    observer = Observer(0, TIME_END / NB_FRAMES, TIME_END, RESOLUTION, DT)
    wall = Component2D("wall", CP, DENSITY, K, THICKNESS, INIT_WALL_TEMPERATURE, air_interior, air_exterior, observer)
    ite_observation0 = (int)(TIME_END / NB_FRAMES / DT)
    assert observer.is_updated(ite_observation0) is True
    assert observer.is_updated(ite_observation0 - 1) is False
    assert observer.is_updated(ite_observation0 + 1) is False
    component_to_solve_list = [wall]
    solver = Solver(component_to_solve_list, fd_transport, DT, TIME_END)
    solver.run()
    # solver.post()
    assert observer.update_count == NB_FRAMES


