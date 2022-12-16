#%% Atmospheric Model

#%% Main Class

class Atmosphere(object):
    
    def __init__(self, gravitational_strength):
    
        # Simulation Atmospheric and Thermodynamic Parameters
        self.g = gravitational_strength # Gravitational Strength