import sys
import os
import numpy as np
from pytest import approx
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from Component import ConstantComponent, Material, Component
from Solver import Solver, Observer, FiniteDifferenceTransport, FiniteVolume, Output

T0 = 273.15 + 25.
EXTERIOR_TEMPERATURE = 273.15 + 33.
INTERIOR_TEMPERATURE = T0
air_exterior = ConstantComponent(EXTERIOR_TEMPERATURE)
air_interior = ConstantComponent(INTERIOR_TEMPERATURE)
CP_AIR = 1000.
DENSITY_AIR = 1.2
K_AIR = 0.025 * 100
air = Material(CP_AIR, DENSITY_AIR, K_AIR)

# build a room component, as a 1D component of thickness BOX_WIDTH.
RESOLUTION = 10
BOX_DEPTH = 1.
BOX_WIDTH = 2.
BOX_HEIGHT = 2.5
DX = BOX_WIDTH / (RESOLUTION)
DT = 0.9 * DX**2 / (2 * (K_AIR / (DENSITY_AIR * CP_AIR)))

TIME_START = 0.
TIME_END = 1 * 3600.
NB_FRAMES = 5
OBSERVER_PERIOD = (int)(TIME_END / NB_FRAMES)

INIT_AIR_TEMPERATURE = np.ones((RESOLUTION)) * T0

neighbours = {'in': air_interior, 'ext': air_exterior}

output_temperature = Output(int(RESOLUTION / 2), var_name='temperature')
# Does not work (plot is the same as raw temperature)
output_temperature_space_avg = Output(0, var_name='temperature', spatial_type='mean')
output_gradient_ext = Output(0, var_name='temperature_gradient', loc='ext')
output_gradient_in = Output(0, var_name='temperature_gradient', loc='in')

def test_observer():
    observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature])
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT, observer=observer)
    room.set_neighbours(neighbours)

    # TODO check extreme setup. Especially the case of a single frame (typically the last one).
    room.observer.set_frame_ite(DT)
    observed_ite = [1]
    for i in range(1,NB_FRAMES):
        ite_observation = (int)(i * (int)(OBSERVER_PERIOD / DT))
        observed_ite.append(ite_observation)
    assert (observed_ite == observer.ite_extraction).all()

    for i in range(1,NB_FRAMES):
        ite_observation = (int)(i * (int)(OBSERVER_PERIOD / DT))
        observed_ite.append(ite_observation)
        print(ite_observation)
        assert observer.is_updated(ite_observation) is True

    component_to_solve_list = [room]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    assert observer.update_count == NB_FRAMES

def test_raw_output():
    observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature, output_gradient_in, output_gradient_ext, output_temperature_space_avg])
    # observer = Observer(TIME_START, OBSERVER_PERIOD, TIME_END, [output_temperature_space_avg])
    room = Component('room', air, BOX_WIDTH, INIT_AIR_TEMPERATURE, FiniteDifferenceTransport(), resolution=RESOLUTION, surface=BOX_DEPTH*BOX_HEIGHT, observer=observer)
    room.set_neighbours(neighbours)
    component_to_solve_list = [room]
    solver = Solver(component_to_solve_list, DT, TIME_END)
    solver.run()
    solver.post()

