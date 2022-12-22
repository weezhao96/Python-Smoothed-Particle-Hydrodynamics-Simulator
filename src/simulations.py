# Simulation

#%% Import

import numpy as np
import math

from functools import reduce
from precision_enums import IntType, FloatType


#%% Simulation Parameters

class SimulationParameter(object):

    # Type Annotation
    n_dim: int
    T: float
    dt: float
    t: float
    float_prec: FloatType
    int_prec: IntType
    
    def __init__(self, n_dim: int,
                 sim_duration: float, dt: float = 0.1,
                 float_precision: FloatType = FloatType.FLOAT64,
                 int_precision: IntType = IntType.INT32):
    
        # Simulation Parameters
        self.n_dim = n_dim # No. of Spatial Dimension
        self.T = sim_duration # Simulation Duration
        self.dt = dt # Simulation Time Step
        self.t = 0.0 # Current Simulation Time
        self.float_prec = float_precision
        self.int_prec = int_precision 
            
        
#%% Simulation Domain

class SimulationDomain(object):

    # Type Annotation
    domain: list[list[float]]
    init_pos: list[list[float]]
    
    def __init__(self, bounding_box:list[list[float]] , initial_position:list[list[float]]):
    
        # Simulation Domain
        self.domain = bounding_box
        
        # Initial Particle Position
        self.init_pos = initial_position
        
        
    def compute_no_of_particles(self, lattice_distance: float, n_dim: int) -> tuple[int, list[int]]:
        
        # Bound Checking
        for dim in range(n_dim):
            
            if self.init_pos[dim][0] <= self.domain[dim][0]:
                self.init_pos[dim][0] = lattice_distance
                
            if self.init_pos[dim][1] >= self.domain[dim][1]:
                self.init_pos[dim][1] = self.init_pos[dim][1] - lattice_distance
    
        
        # Compute Particle per Dimension
        n_per_dim = []
        
        for dim in range(n_dim):
            
            domain_length = self.init_pos[dim][1] - self.init_pos[dim][0] + np.finfo(float).eps    
            n_per_dim.append(int(math.floor(domain_length / lattice_distance)) + 1)
            

        n_particle_G = reduce(lambda x, y: x * y, n_per_dim)
        
        return n_particle_G, n_per_dim
        