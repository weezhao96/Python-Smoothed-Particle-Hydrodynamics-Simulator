#%% Simulation

from sph import BaseSPH

#%% Simulation Parameters

class SimulationParameter(object):
    
    def __init__(self, sim_duration, n_dim, dt = None):
    
        # Simulation Parameters - Time
        self.T = sim_duration # Simulation Duration
        self.dt = dt # Simulation Time Step
        self.t = 0 # Current Simulation Time
        self.n_dim = n_dim # No. of Spatial Dimension
        
        
#%% Simulation Domain

class SimulationDomain(object):
    
    def __init__(self, simulation_domain, initial_position):
    
        # Simulation Domain
        self.simulation_domain = simulation_domain
        
        # Initial Particle Position
        self.init_pos = initial_position
        
        
    def compute_initial_position(self, sph):
        
        n_dim = sph.simulation_param.n_dim
        
        n_particle = 0
        