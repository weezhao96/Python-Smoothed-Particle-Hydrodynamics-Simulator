# Smoothed Particle Hydrodynamics Model

#%% Import

import numpy as np
import abc

from mp_manager import MP_Manager
from io_manager import IO_Manager
from precision_enum import IntType, FloatType


#%% Main Class

class BaseSPH(object):
    
    def __init__(self, atmospheric_model, particle_model,
                 sim_param, sim_domain, kernel,
                 n_process, output_path):
        
        # Model
        self.atmospheric_model = atmospheric_model
        self.particle_model = particle_model
        
        # Simulation
        self.sim_param = sim_param
        self.sim_domain = sim_domain
        
        # Kernel
        self.kernel = kernel
        
        # Manager
        self.mp_manager = MP_Manager(n_process)
        self.io_manager = IO_Manager(output_path)
        
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
        
        self.n_particle_G = self.sim_domain.compute_no_of_particles()
        self.mp_manager.assign_share_memory(self.n_particle_G,
                                            self.sim_param)
        self.__init_particle_state()

        
        
        
    #%% Simulation Algorithm
    
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
        self.Ep = np.ndarray(self.n_particle, dtype = np.float64, buffer=self.x, 
                             offset=self.sim_param.n_dim, strides=self.sim_param.n_dim)
        
        # Energy Computation
        m = self.particle_model.m
        self.Ek = 0.5 * m * self.Ek
        self.Ep = m * self.atmospheric_model.g * self.Ep
        
        # Total Energy Computation
        self.Ek_total = np.sum(self.Ek)
        self.Ep_total = np.sum(self.Ep)

        
    #%% Simulation Utilities
        
    def __init_particle_state(self):
        
        # Array Shape
        shape_2D = self.n_particle_G * self.sim_param.n_dim
        shape_1D = self.n_particle_G
        
        # Data Type
        dtype_float = self.sim_param.float_precision.get_np_dtype()
        dtype_int = self.sim_param.int_precision.get_np_dtype()
        
        # State Allocation
        self.x_G = np.ndarray(shape_2D, dtype = dtype_float,
                              buffer=self.mp_manager.shm['x_G'].buf)
        self.x_G = np.array([0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5,
                             0.5, 0.5, 0.6, 0.5, 0.7, 0.5, 0.8, 0.5, 0.9, 0.5])
        
        self.v_G = np.ndarray(shape_2D, dtype = dtype_float,
                              buffer=self.mp_manager.shm['v_G'].buf)
        self.v_G = np.array([0 for i in range(self.n_particle_G)]) 
        
        self.a_G = np.ndarray(shape_2D, dtype = dtype_float,
                              buffer=self.mp_manager.shm['a_G'].buf)
        self.a_G = np.array([0 for i in range(self.n_particle_G)])
        
        self.id_G = np.ndarray(shape_1D, dtype = dtype_int,
                               buffer=self.mp_manager.shm['id_G'].buf)
        self.id_G = np.array([i for i in range(self.n_particle_G)])
        
        self.rho_G = np.ndarray(shape_1D, dtype = dtype_float,
                                buffer=self.mp_manager.shm['rho_G'].buf)
        self.rho_G = np.array([1000.0 for i in range(self.n_particle_G)])
        
        self.p_G = np.ndarray(shape_1D, dtype = dtype_float,
                              buffer=self.mp_manager.shm['p_G'].buf)
        self.p_G = np.array([1000.0 for i in range(self.n_particle_G)])