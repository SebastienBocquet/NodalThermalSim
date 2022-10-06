# NodalThermalSim

This tool runs simulations of unsteady thermal phenomenon in time for a system of connected components.
Components can be:
  - a thin layer of material (typically a wall), in which the 1D unsteady heat equation is solved.
    Supported boundary conditions include Dirichlet temperature, fixed heat flux and fixed heat transfer coefficient.
  - a box, in which a single temperature point is solved. Its value is updated based on energy conservation
    (typically the temperature variation is determined by the sum of fluxes through the box faces).

Time varying heat sources can be taken into account.

