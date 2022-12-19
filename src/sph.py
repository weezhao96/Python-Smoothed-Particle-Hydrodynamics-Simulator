# Smoothed Particle Hydrodynamics Model

#%% Import

import numpy as np
import abc
import matplotlib.pyplot as plt

from mp_manager import MP_Manager
from io_manager import IO_Manager


#%% Base Class Definition

class BaseSPH(object):
    
    def __init__(self, utility, atmospheric_model, particle_model,
                 sim_param, sim_domain, kernel,
                 n_process, output_path):
        
        # Utility Module
        self.util = utility
        
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
        
    
    #%% Simulation Algorithm
    
    @abc.abstractmethod
    def run_simulation(self):
        pass
    
    @abc.abstractmethod
    def __boundary_check(self):
        pass
    
    def __map_neighbour(self):
        pass
    
    def __density_pressure_computation(self):
        pass
        
    def __rescale_mass_density_pressure(self):
        pass
    
    @abc.abstractmethod
    def __accel_computation(self):
        pass
    
    @abc.abstractmethod
    def __time_stepping(self):
        pass
    
    def __energy_computation(self):
        
        n_dim = self.sim_param.n_dim
        
        # Loop (Partial Energy Computation)
        for i in range(self.n_particle):
            
            # Index
            index_start = (i-1) * n_dim
            index_end = index_start + n_dim
            
            # Ek <- v.v
            v = self.v[index_start:index_end]
            self.Ek[i] = np.dot(v,v)
            
        
        # Ep <- z
        self.Ep = np.ndarray(self.n_particle, dtype=np.float64, buffer=self.x, 
                             offset=self.sim_param.n_dim, strides=self.sim_param.n_dim)
        
        # Energy Computation
        m = self.particle_model.m
        self.Ek = 0.5 * m * self.Ek
        self.Ep = m * self.atmospheric_model.g * self.Ep
        
        # Total Energy Computation
        self.Ek_total = np.sum(self.Ek)
        self.Ep_total = np.sum(self.Ep)
        

#%% BasicSPH

class BasicSPH(BaseSPH):
    
    def run_simulation(self):
        
        self.n_particle_G = self.sim_domain.compute_no_of_particles()
        self.mp_manager.assign_share_memory(self.n_particle_G, self.sim_param)
        self.util.init_particle_state(self)
        
        self.__boundary_check()

        fig, ax = plt.subplots()
        
        while (self.sim_param.t < self.sim_param.T - 0.00000001):
            
            self.__accel_computation()
            self.__time_stepping()
            self.__boundary_check()
    
            self.sim_param.t += self.sim_param.dt
            
            self.util.plot(self, plt, ax)
            
            
            
    def __boundary_check(self):
        
        # Index
        index = 0
        
        # Bounding Box
        box = self.sim_domain.bounding_box
        
        # Loop
        for i in range(self.n_particle_G):
            for dim in range(self.sim_param.n_dim):
                
                # Lower Bound
                if (self.x[index] < box[dim][0] + self.particle_model.h):
                    
                    self.x[index] = box[dim][0] + self.particle_model.h
                    self.v[index] *= -0.5
                    
                elif (self.x[index] > box[dim][1] - self.particle_model.h):
                    
                    self.x[index] = box[dim][1] - self.particle_model.h
                    self.v[index] *= -0.5
                    
                index += 1
                
    
    def __accel_computation(self):
        
        self.a = 1.0 * np.random.rand(self.n_particle_G * self.sim_param.n_dim) - 0.5
        

    def __time_stepping(self):
        
        self.v = self.v + self.a * self.sim_param.dt
        self.x = self.x + self.v * self.sim_param.dt
        