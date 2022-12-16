#%% Particle Model

#%% Main Class

class Particle(object):
    
    def __init__(self, radius_of_influence,
                 resting_density, viscosity,
                 specific_heat_ratio, speed_of_sound):
    
        # Simulation Particle Parameters
        self.h = radius_of_influence # Particle Radius of Influence
        self.rho_0 = resting_density # Resting Density
        self.mu = viscosity # Viscosity
        self.gamma = specific_heat_ratio
        self.c = speed_of_sound
        self.m = 1 # Particle Mass = 1 - assumed to be 1, and rescaled during start of simulation