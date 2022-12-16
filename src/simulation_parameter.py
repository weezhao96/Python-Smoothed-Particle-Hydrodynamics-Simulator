#%% Simulation Parameters

#%% Main Class

class SimulationParameter(object):
    
    def __init__(self, sim_duration, n_dim, dt = None):
    
        # Simulation Parameters - Time
        self.T = sim_duration # Simulation Duration
        self.dt = dt # Simulation Time Step
        self.t = 0 # Current Simulation Time
        self.n_dim = n_dim # No. of Spatial Dimension