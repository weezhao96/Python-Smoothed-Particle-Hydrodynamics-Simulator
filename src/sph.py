# Smoothed Particle Hydrodynamics Model

#%% Import

import numpy as np
import abc
import matplotlib.pyplot as plt

from sph_util import SPH_Util
from models import Atmosphere, Particle
from simulations import SimulationParameter, SimulationDomain
from kernels import BaseKernel, QuinticKernel
from precision_enums import IntType, FloatType
from mp_manager import MP_Manager
from io_manager import IO_Manager
from interaction import Interaction


#%% Base Class Definition

class BaseSPH(object):

    # Type Annotation
    util: SPH_Util

    atmospheric_model: Atmosphere
    particle_model: Particle

    sim_param: SimulationParameter
    sim_domain: SimulationDomain

    kernel: BaseKernel

    mp_manager: MP_Manager
    io_manager: IO_Manager

    n_particle_G: int
    id_G: np.ndarray
    x_G: np.ndarray
    v_G: np.ndarray
    a_G: np.ndarray
    rho_G: np.ndarray
    p_G: np.ndarray

    n_particle: int
    id: np.ndarray
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    rho: np.ndarray
    p: np.ndarray

    Ek_G: np.ndarray
    Ep_G: np.ndarray
    E_G: np.ndarray
    Ek_total_G: np.float_
    Ep_total_G: np.float_
    E_total_G: np.float_

    Ek: np.ndarray
    Ep: np.ndarray
    E: np.ndarray
    Ek_total: np.float_
    Ep_total: np.float_
    E_total: np.float_
    
    def __init__(self, utility: SPH_Util,
                 atmospheric_model: Atmosphere, particle_model: Particle,
                 sim_param: SimulationParameter, sim_domain: SimulationDomain,
                 kernel: BaseKernel,
                 mp_manager: MP_Manager, io_manager: IO_Manager):
        
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
        self.mp_manager = mp_manager
        self.io_manager = io_manager
        
        # Global State Variables
        self.n_particle_G = int() # No. of Partiles
        self.id_G = np.array([]) # Particle ID
        self.x_G = np.array([]) # Particle Position
        self.v_G = np.array([]) # Particle Velocity
        self.a_G = np.array([]) # Particle Acceleration
        self.rho_G = np.array([]) # Particle Density
        self.p_G = np.array([]) # Particle Pressure
        
        # Local State Variables
        self.n_particle = int() # No. of Partiles
        self.id = np.array([]) # Particle ID
        self.x = np.array([]) # Particle Position
        self.v = np.array([]) # Particle Velocity
        self.a = np.array([]) # Particle Acceleration
        self.rho = np.array([]) # Particle Density
        self.p = np.array([]) # Particle Pressure
        
        # Global Energy Variables
        self.Ek_G = np.array([]) # Particle Kinetic Energy
        self.Ep_G = np.array([]) # Particle Potential Energy
        self.E_G = np.array([]) # Particle Total Energy
        self.Ek_total_G = np.float_() # Total Kinetic Energy
        self.Ep_total_G = np.float_() # Total Potential Energy
        self.E_total_G = np.float_() # Total Energy
        
        # Local Energy Variables
        self.Ek = np.array([]) # Particle Kinetic Energy
        self.Ep = np.array([]) # Particle Potential Energy
        self.E = np.array([]) # Particle Total Energy
        self.Ek_total = np.float_() # Total Kinetic Energy
        self.Ep_total = np.float_() # Total Potential Energy
        self.E_total = np.float_() # Total Energy
        
    
    #%% Simulation Algorithm
    
    @abc.abstractmethod
    def run_simulation(self):
        pass
    
    @abc.abstractmethod
    def _boundary_check(self):
        pass
    
    @abc.abstractmethod
    def _map_neighbour(self):
        pass
    
    @abc.abstractmethod
    def _density_pressure_computation(self):
        pass
        
    def _rescale_mass_density_pressure(self):
                
        self.particle_model.m = 1.0
    
    @abc.abstractmethod
    def _accel_computation(self):
        pass
    
    @abc.abstractmethod
    def _time_stepping(self):
        pass
    
    def _energy_computation(self):
        
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
        self.Ep = self.x[self.sim_param.n_dim-1::self.sim_param.n_dim]

        # Energy Computation
        m = self.particle_model.m
        self.Ek = 0.5 * m * self.Ek
        self.Ep = m * self.atmospheric_model.g * self.Ep
        self.E = self.Ek + self.Ep
        
        # Total Energy Computation
        self.Ek_total = np.sum(self.Ek)
        self.Ep_total = np.sum(self.Ep)
        self.E_total = np.sum(self.E)
        
        
    def clean_up_simulation(self):
        
        for i in self.mp_manager.shm:
            
            self.mp_manager.shm[i].close()
            self.mp_manager.shm[i].unlink()
        

#%% BasicSPH

class BasicSPH(BaseSPH):
    
    def run_simulation(self):
        
        # Instantiate Particle State and Distribute to Process
        self.util.init_particle_state(self)

        for attr in self.__dict__:

            if attr[-2:] == '_G':
                self.__dict__[attr[:-2]] = self.__dict__[attr]

        # Perturb Particle
        self.util.perturb_particle(self)        
        self._boundary_check()
        
        # Map Neighbour
        interaction_set = []

        for i in range(self.n_particle):

            new_interaction = Interaction(self.id[i], self.util.kissing_num, self.sim_param)
            interaction_set.append(new_interaction)

        self._map_neighbour(interaction_set)

        # Density Pressure Computation
        self._density_pressure_computation()
        self._rescale_mass_density_pressure()

        # Acceleration Computation
        self._accel_computation()

        # Energy Computation
        self._energy_computation()

        # Output Results
        fig, ax = plt.subplots()
        self.util.plot(self, plt, ax)

        print('t = {:.3f}'.format(self.sim_param.t))
        print('Total Kinetic Energy = {:.3f}'.format(self.Ek_total))
        print('Total Potential Energy = {:.3f}'.format(self.Ep_total))
        print(' ')

        # First Timestepping
        self._first_time_stepping()
        self._boundary_check()

        self.sim_param.t += self.sim_param.dt

        # Timestep Looping            
        while (self.sim_param.t < self.sim_param.T - np.finfo(float).eps):
            
            # Map Neighbour
            self._map_neighbour(interaction_set)

            # Density Pressure Computation
            self._density_pressure_computation()

            # Acceleration Computation
            self._accel_computation()
            
            # Energy Computation
            self._energy_computation()
            
            # Output Results
            fig, ax = plt.subplots()
            self.util.plot(self, plt, ax)

            print('t = {:.3f}'.format(self.sim_param.t))
            print('Total Kinetic Energy = {:.3f}'.format(self.Ek_total))
            print('Total Potential Energy = {:.3f}'.format(self.Ep_total))
            print(' ')

            # Time Stepping
            self._time_stepping()
            self._boundary_check()
    
            self.sim_param.t += self.sim_param.dt


        self.clean_up_simulation()
            
            
    def _boundary_check(self):
        
        # Index
        index = 0
        
        # Bounding Box
        domain = self.sim_domain.domain
        boundary_radius = self.kernel.radius_of_influence * self.kernel.h
        
        # Loop
        for i in range(self.n_particle_G):
            for dim in range(self.sim_param.n_dim):
                
                # Lower Bound
                if (self.x[index] <  domain[dim][0] + boundary_radius):
                    
                    self.x[index] =  domain[dim][0] + boundary_radius
                    self.v[index] *= -0.5
                    
                elif (self.x[index] >  domain[dim][1] - boundary_radius):
                    
                    self.x[index] =  domain[dim][1] - boundary_radius
                    self.v[index] *= -0.5
                    
                index += 1
                
    
    def _map_neighbour(self, inter_set: list[Interaction]):
        
        n_dim = self.sim_param.n_dim

        # Reset Interaction
        for i in range(self.n_particle):
            inter_set[i].reset()
            
        # Compute Interaction

        for i in range(self.n_particle):

            id_i = self.id[i]
            index_i = i * n_dim

            for j in range(i+1, self.n_particle):

                id_j = self.id[j]
                index_j = j * n_dim

                dr = self.x[index_i:index_i+n_dim] - self.x[index_j:index_j+n_dim]
                q = np.linalg.norm(dr, ord=2) / self.kernel.h

                if q < self.kernel.radius_of_influence:
                    
                    dv = self.v[index_i:index_i+n_dim] - self.v[index_j:index_j+n_dim]

                    inter_set[i].add_neighbour(id_j, q, dr, dv)
                    inter_set[j].add_neighbour(id_i, -q, -dr, -dv)



    def _accel_computation(self):
        
        n_dim = self.sim_param.n_dim
        shape = self.a.shape[0]
        
        self.a = 1.0 * np.random.rand(shape).astype(self.sim_param.float_prec.get_np_dtype()) - 0.5

        # self.a[n_dim-1:shape:n_dim] = self.a[n_dim-1:shape:n_dim] - self.atmospheric_model.g


    def _first_time_stepping(self):
        pass


    def _time_stepping(self):
        
        self.v = self.v + self.a * self.sim_param.dt
        #self.v = 1.0 * np.random.rand(self.n_particle * self.sim_param.n_dim) - 0.5
        self.x = self.x + self.v * self.sim_param.dt
