# Smoothed Particle Hydrodynamics Model

#%% Import

from sph_util import SPH_Util
from models import Atmosphere, Particle
from simulations import SimulationParameter, SimulationDomain
from kernels import BaseKernel, QuinticKernel
from precision_enums import IntType, FloatType
from mp_manager import MP_Manager
from io_manager import IO_Manager
from interaction import Interaction

import numpy as np
import abc


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

        self.particle_model.m = self.n_particle * self.particle_model.rho_0 / np.sum(self.rho)

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
        self.Ek = 0.5 * self.particle_model.m * self.Ek
        self.Ep = self.particle_model.m * self.atmospheric_model.g * self.Ep
        self.E = self.Ek + self.Ep
        
        # Total Energy Computation
        self.Ek_total = np.sum(self.Ek)
        self.Ep_total = np.sum(self.Ep)
        self.E_total = np.sum(self.E)
    

    def _aggregate_total_energy(self):

        self.Ek_total_G = self.Ek_total
        self.Ep_total_G = self.Ep_total
        self.E_total_G = self.Ek_total_G + self.Ep_total_G
        

    def _sync_L2G(self):

        # Output Result
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.rho, self.rho_G, 1)
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.p, self.p_G, 1)

        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.x, self.x_G, 2)
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.v, self.v_G, 2)
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.a, self.a_G, 2)

        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.E, self.E_G, 1)
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.Ek, self.Ek_G, 1)
        self.mp_manager.comm_L2G(self.sim_param.n_dim, self.n_particle, self.id,
                                 self.Ep, self.Ep_G, 1)

        self._aggregate_total_energy()


    def clean_up_simulation(self):
        
        self.io_manager.state_writer.kill_writer_thread(self.n_particle_G)
        self.io_manager.energy_writer.kill_writer_thread(self.n_particle_G)

        for i in self.mp_manager.shm:
            
            self.mp_manager.shm[i].close()
            self.mp_manager.shm[i].unlink()
        

#%% BasicSPH

class BasicSPH(BaseSPH):
    
    def run_simulation(self):
        
        # Instantiate Particle State and Distribute to Process
        self.util.init_particle_global_states(self)

        for attr in self.__dict__:

            if attr[-2:] == '_G':
                self.__dict__[attr[:-2]] = self.__dict__[attr]

        # Perturb Particle
        self.util.perturb_particles(self)        
        self._boundary_check()
        
        # Map Neighbour
        inter_set = []

        for i in range(self.n_particle):

            new_interaction = Interaction(self.id[i], self.util.kissing_num, self.sim_param)
            inter_set.append(new_interaction)

        self._map_neighbour(inter_set)

        # Density Pressure Computation
        self._density_pressure_computation(inter_set)
        self._rescale_mass_density_pressure()
        self._density_pressure_computation(inter_set)

        # Acceleration Computation
        self._accel_computation(inter_set)

        # Energy Computation
        self._energy_computation()                        

        # Output Result
        self._sync_L2G()

        self.io_manager.state_writer.output_data(self.n_particle_G, self.sim_param.n_dim, self.sim_param.t, self.sim_param.t_count,
                                                 self.x_G, self.v_G, self.a_G, self.rho_G, self.p_G)
        self.io_manager.energy_writer.output_data(self.sim_param.t, self.Ek_total_G, self.Ep_total_G, self.E_total_G)                                                    

        print('t = {:.3f}'.format(self.sim_param.t))
        print('Total Kinetic Energy = {:.3f}'.format(self.Ek_total_G))
        print('Total Potential Energy = {:.3f}'.format(self.Ep_total_G))
        print(' ')

        # First Timestepping
        self._first_time_stepping()
        self._boundary_check()

        self.sim_param.t += self.sim_param.dt
        self.sim_param.t_count += 1

        # Timestep Looping            
        while (self.sim_param.t < self.sim_param.T - np.finfo(float).eps):
            
            # Map Neighbour
            self._map_neighbour(inter_set)

            # Density Pressure Computation
            self._density_pressure_computation(inter_set)

            # Acceleration Computation
            self._accel_computation(inter_set)

            # Energy Computation
            self._energy_computation()

            # Output Results
            if self.sim_param.t_count % 10 == 0:

                self.io_manager.state_writer.sync_queue(self.n_particle_G)

                self._sync_L2G()
               
                self.io_manager.state_writer.output_data(self.n_particle_G, self.sim_param.n_dim, self.sim_param.t, self.sim_param.t_count,
                                                         self.x_G, self.v_G, self.a_G, self.rho_G, self.p_G)
                                                               
                self.io_manager.energy_writer.output_data(self.sim_param.t, self.Ek_total_G, self.Ep_total_G, self.E_total_G)                                                                                                            
                
                print('t = {:.3f}'.format(self.sim_param.t))
                print('Total Kinetic Energy = {:.3f}'.format(self.Ek_total_G))
                print('Total Potential Energy = {:.3f}'.format(self.Ep_total_G))
                print(' ')


            # Time Stepping
            self._time_stepping()
            self._boundary_check()
    
            self.sim_param.t += self.sim_param.dt
            self.sim_param.t_count += 1

        # Output Result
        self.io_manager.state_writer.sync_queue(self.n_particle_G)

        self._sync_L2G()
        
        self.io_manager.state_writer.output_data(self.n_particle_G, self.sim_param.n_dim, self.sim_param.t, self.sim_param.t_count,
                                                 self.x_G, self.v_G, self.a_G, self.rho_G, self.p_G)
        self.io_manager.energy_writer.output_data(self.sim_param.t, self.Ek_total_G, self.Ep_total_G, self.E_total_G)                                                                                                            
        
        print('t = {:.3f}'.format(self.sim_param.t))
        print('Total Kinetic Energy = {:.3f}'.format(self.Ek_total))
        print('Total Potential Energy = {:.3f}'.format(self.Ep_total))
        print(' ')

        self.clean_up_simulation()
            
            
    def _boundary_check(self):
        
        # Index
        index = 0
        
        # Bounding Box
        domain = self.sim_domain.domain
        boundary_radius = self.kernel.radius_of_influence * self.kernel.h
        
        # Loop
        for i in range(self.n_particle):
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

                dr = self.x[index_i : index_i+n_dim] - self.x[index_j : index_j+n_dim]
                q = np.linalg.norm(dr, ord=2) / self.kernel.h

                if q < self.kernel.radius_of_influence:
                    
                    dv = self.v[index_i : index_i+n_dim] - self.v[index_j : index_j+n_dim]

                    inter_set[i].add_neighbour(id_j, q, dr, dv)
                    inter_set[j].add_neighbour(id_i, q, -dr, -dv)


    def _density_pressure_computation(self, inter_set: list[Interaction]):

        # Density
        for i in range(self.n_particle):
            
            # Kernel
            W = self.kernel.W(inter_set[i].q[:inter_set[i].index_1D])

            # Density
            self.rho[i] = np.sum(W)

        self.rho = self.rho + self.kernel.W(np.array([0.0], dtype=self.sim_param.float_prec.get_np_dtype()))
        self.rho = self.particle_model.m * self.rho

        # Pressure
        k = self.particle_model.c ** 2 * self.particle_model.rho_0 / self.particle_model.gamma
        self.p = k * (np.power((self.rho / self.particle_model.rho_0), self.particle_model.gamma) - 1.0) 


    def _accel_computation(self, inter_set: list[Interaction]):
        
        # Parameter
        n_dim = self.sim_param.n_dim
        shape = self.a.shape[0]
        
        # Pressure Computation
        a_p = np.zeros(shape=shape, dtype=self.sim_param.float_prec.get_np_dtype())

        for i in range(self.n_particle):
            
            index_1D_i = i
            index_2D_i = i * n_dim

            p_rho_i = self.p[index_1D_i] / self.rho[index_1D_i] ** 2

            for j in range(inter_set[i].n_neighbour):
                
                # 1D and 2D Index
                id_L_j = inter_set[i].id_L_neighbour[j]

                # Pressure
                q = inter_set[i].q[j]

                nabla_W = self.kernel.nabla_W(q)

                p_scale = -self.particle_model.m * nabla_W
                p_rho = p_rho_i + self.p[id_L_j] / self.rho[id_L_j] ** 2
                unit_vec = inter_set[i].dr[j * n_dim : (j+1) * n_dim] / (q * self.kernel.h)

                a_p[index_2D_i : index_2D_i + n_dim] = a_p[index_2D_i : index_2D_i + n_dim] + unit_vec * p_rho * p_scale

        self.a = a_p

        # Gravitational Forcing
        self.a[n_dim-1::n_dim] = self.a[n_dim-1::n_dim] - self.atmospheric_model.g


    def _first_time_stepping(self):
        
        self.v = self.v + 0.5 * self.sim_param.dt * self.a
        self.x = self.x + self.sim_param.dt * self.v


    def _time_stepping(self):
        
        self.v = self.v + self.a * self.sim_param.dt
        self.x = self.x + self.v * self.sim_param.dt