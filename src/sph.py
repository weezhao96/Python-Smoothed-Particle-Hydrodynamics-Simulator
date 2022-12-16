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
        
        # Kernal
        self.kernel = kernel
        
        # Manager
        self.mp_manager = mp_manager.MP_Manager(n_process)
        self.io_manager = io_manager.IO_Manager(output_path)
        
        # # Global State Variables
        # self.n_particle_G = None # No. of Partiles
        # self.id_G = None # Particle ID
        # self.x_G = None # Particle Position
        # self.v_G = None # Particle Velocity
        # self.a_G = None # Particle Acceleration
        # self.rho_G = None # Particle Density
        # self.p_G = None # Particle Pressure
        
        # Global State Variables
        self.n_particle_G = 6 # No. of Partiles
        self.id_G = np.array([0,1,2,3,4,5]) # Particle ID
        self.x_G = np.array([0,1,2,3,4,5]) # Particle Position
        self.v_G = np.array([0,1,2,3,4,5]) # Particle Velocity
        self.a_G = np.array([0,1,2,3,4,5]) # Particle Acceleration
        self.rho_G = np.array([0,1,2,3,4,5]) # Particle Density
        self.p_G = np.array([0,1,2,3,4,5]) # Particle Pressure
        
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
        
    def __PerturbParticle(self):
        pass
    
    def __BoundaryCheck(self):
        pass
    
    def __MapNeighbour(self):
        pass
    
    def __DensityPressureComputation(self):
        pass
        
    def __RescaleMassDensityPressure(self):
        pass
    
    def __AccelComputation(self):
        pass
    
    def __EnergyComputation(self):
        
        
        
        
        
        
        
