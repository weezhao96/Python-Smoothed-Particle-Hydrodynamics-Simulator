# Simulation

#%% Import

from dataclasses import dataclass
from precision_enums import IntType, FloatType
from kernels import BaseKernel
from functools import reduce

import numpy as np


#%% Simulation Parameters

@dataclass
class SimulationParameter(object):

    # Type Annotation
    n_dim: int
    T: float
    dt: float
    float_prec: FloatType
    int_prec: IntType
    t: float = 0.0
    t_count: int = 0

            
        
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
        
        
    def compute_no_of_particles(self, sim_param: SimulationParameter, kernel: BaseKernel) -> tuple[int, tuple[int, ...]]:
        
        # Variables
        n_dim = sim_param.n_dim
        lattice_distance = kernel.radius_of_influence * kernel.h

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
            n_per_dim.append(int(np.floor(domain_length / lattice_distance)) + 1)
        
        n_per_dim = tuple(n_per_dim)

        n_particle_G = reduce(lambda x, y: x * y, n_per_dim)
        
        return n_particle_G, n_per_dim
        