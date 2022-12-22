# Model

#%% Atmospheric Model

class Atmosphere(object):

    # Type Annotation
    g: float
    
    def __init__(self, gravitational_strength: float):
    
        # Simulation Atmospheric and Thermodynamic Parameters
        self.g = gravitational_strength # Gravitational Strength
        
        
#%% Particle Model

class Particle(object):

    # Type Annotation
    rho_0: float
    mu: float
    gamma: float
    c: float
    m: float
    
    def __init__(self, resting_density: float, viscosity: float,
                 specific_heat_ratio: float, speed_of_sound: float):
    
        # Simulation Particle Parameters
        self.rho_0 = resting_density # Resting Density
        self.mu = viscosity # Viscosity
        self.gamma = specific_heat_ratio # Specific Heat Ratio
        self.c = speed_of_sound # Speed of Sound
        self.m = 1.0 # Particle Mass = 1 - assumed to be 1, and rescaled during start of simulation