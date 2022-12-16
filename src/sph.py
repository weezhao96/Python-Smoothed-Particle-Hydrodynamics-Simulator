#%% Smoothed Particle Hydrodynamics Model

#%% Import

import numpy as np
import abc

import mp_manager, io_manager

#%% Main Class

class BaseSPH(object):
    
    def __init__(self, atmospheric_model, particle_model,
                 sim_param, sim_domain, kernel,
                 n_process, output_path):
        
        # Model
        self.atmospheric_model = atmospheric_model
        self.particle_model = particle_model
        
        # Simulation
        self.simulation_param = sim_param
        self.simulation_domain = sim_domain
        
        # Kernel
        self.kernel = kernel
        
        # Manager
        self.mp_manager = mp_manager.MP_Manager(n_process)
        self.io_manager = io_manager.IO_Manager(output_path)
        
        # Global State Variables
        self.n_particle_G = None # No. of Partiles
        self.id_G = None # Particle ID
        self.x_G = None # Particle Position
        self.v_G = None # Particle Velocity
        self.a_G = None # Particle Acceleration
        self.rho_G = None # Particle Density
        self.p_G = None # Particle Pressure
        
        # Local State Variables
        self.n_particle = None
        self.id = None # Particle ID
        self.x = None # Particle Position
        self.v = None # Particle Velocity
        self.a = None # Particle Acceleration
        self.rho = None # Particle Density
        self.p = None # Particle Pressure
        
        # Global Energy Variables
        self.Ek_G = None # Particle Kinetic Energy
        self.Ep_G = None # Particle Potential Energy
        self.E_G = None # Particle Total Energy
        self.Ek_total_G = None # Total Kinetic Energy
        self.Ep_total_G = None # Total Potential Energy
        self.E_total_G = None # Total Energy
        
        # Local Energy Variables
        self.Ek = None # Particle Kinetic Energy
        self.Ep = None # Particle Potential Energy
        self.E = None # Particle Total Energy
        self.Ek_total = None # Total Kinetic Energy
        self.Ep_total = None # Total Potential Energy
        self.E_total = None # Total Energy
        
        
    def run_simulation(self):
        
        self.simulation_domain.init_position(self)
        
        
        
    
    def __perturb_particle(self):
        pass
    
    def __boundary_check(self):
        pass
    
    def __map_neighbour(self):
        pass
    
    def __density_pressure_computation(self):
        pass
        
    def __rescale_mass_density_pressure(self):
        pass
    
    def __accel_computation(self):
        pass
    
    def __energy_computation(self):
        
        n_dim = self.simulation_param.n_dim
        
        # Loop (Partial Energy Computation)
        for i in range(self.n_particle):
            
            # Index
            index_start = (i-1) * n_dim
            index_end = index_start + n_dim
            
            # Ek <- v.v
            v = self.v[index_start:index_end]
            self.Ek[i] = np.dot(v,v)
            
            # Ep <- z
            self.Ep[i] = x[index_end - 1]
        
        # Energy Computation
        m = self.particle_model.m
        self.Ek = 0.5 * m * self.Ek
        self.Ep = m * self.atmospheric_model.g * self.Ep
        
        # Total Energy Computation
        self.Ek_total = np.sum(self.Ek)
        self.Ep_total = np.sum(self.Ep)

        
#%% 
        
        
        
