# Simulation Run

#%% Library

import atmosphere, particle
import simulation_domain, simulation_parameter, kernel
import sph

#%% Model Definition

# Atmosphere
earth = atmosphere.Atmosphere(gravitational_strength = 9.81)

# Particle
water = particle.Particle(radius_of_influence = 0.01,
                          resting_density = 1000.0, viscosity = 1.0,
                          specific_heat_ratio = 7.0, speed_of_sound = 1480.0)

#%% Simulation Definition

sim_param = simulation_parameter.SimulationParameter(sim_duration = 10.0,
                                                     n_dim = 3)

unit_cube = simulation_domain.SimulationDomain(simulation_domain =
                                               [[0.0,0.0,0.0],
                                                [1.0,1.0,1.0]])


quintiq = kernel.QuinticKernel(n_dim = sim_param.n_dim,
                               radius_of_influence = water.h)

#%% SPH

model = sph.BaseSPH(earth, water,
                    sim_param, unit_cube, quintiq,
                    1, "output/")