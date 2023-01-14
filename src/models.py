# Model

#%% Import

from dataclasses import dataclass

#%% Atmospheric Model

@dataclass
class Atmosphere(object):

    # Type Annotation
    g: float
        
        
#%% Particle Model

@dataclass
class Particle(object):

    # Type Annotation
    rho_0: float
    mu: float
    gamma: float
    c: float
    m: float = 1.0
    