Simulate the temporal evolution of a house energy balance for a typical day of the year.

Physics taken into account:

  - radiative solar heat flux, including ray angle and screening (shadow of the roof or any additional surface)
  - heat diffusion through the walls (including the ceiling and the floor)
  - split house in different rooms
  - ventilation

Use input:

  - external air temperature evolution
  - date
  - wall parameters: geometry, orientation, thermal conductivity (k), density, Cp, percentage of screening versus time
  - ventilation mass flow rate versus time
  - external heat flux transfer coef (convection + IR radiation) in $W.m^{-2}.K^{-1}$ 
  - internal heat flux transfer coef (IR radiation)
  - roof parameters: thermal conductivity (k), density, Cp
  - floor heat flux in $W.m^{-2}$
  - initial conditions (interior air temperature)

Output:

  - interior air temperature evolution



