# Simulation

#%% Import

import numpy as np

from sph import BaseSPH
from precision_enum import IntType, FloatType


#%% Simulation Parameters

class SimulationParameter(object):
    
    def __init__(self, n_dim, sim_duration, dt = None,
                 float_precision=FloatType.FLOAT64,
                 int_precision=IntType.INT32):
    
        # Simulation Parameters
        self.n_dim = n_dim # No. of Spatial Dimension
        self.T = sim_duration # Simulation Duration
        self.dt = dt # Simulation Time Step
        self.t = 0 # Current Simulation Time
        self.float_precision = float_precision 
        self.int_precision = int_precision 
            
        
#%% Simulation Domain

class SimulationDomain(object):
    
    def __init__(self, bounding_box, initial_position):
    
        # Simulation Domain
        self.bounding_box = bounding_box
        
        # Initial Particle Position
        self.init_pos = initial_position
        
        
    def compute_no_of_particles(self):
                
        n_particle_G = 9
        
        return n_particle_G
        
        