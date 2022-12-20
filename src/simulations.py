# Simulation

#%% Import

import numpy as np
import math

from functools import reduce
from precision_enums import IntType, FloatType


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
        
        
    def compute_no_of_particles(self, h, n_dim):
        
        # Bound Checking
        for dim in range(n_dim):
            
            if self.init_pos[dim][0] <= self.bounding_box[dim][0]:
                self.init_pos[dim][0] = h
                
            if self.init_pos[dim][1] >= self.bounding_box[dim][1]:
                self.init_pos[dim][1] = self.init_pos[dim][1] - h
    
        
        # Compute Particle per Dimension
        n_per_dim = []
        
        for dim in range(n_dim):
            
            length = self.init_pos[dim][1] - self.init_pos[dim][0] + np.finfo(float).eps    
            n_per_dim.append(int(math.floor(length / h)) + 1)
            

        n_particle_G = reduce(lambda x, y: x * y, n_per_dim)
            
        
        return n_particle_G, n_per_dim
        