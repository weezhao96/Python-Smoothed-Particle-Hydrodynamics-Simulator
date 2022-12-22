# SPH_Util

#%% Import

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sph import BaseSPH


#%% Class Definition

class SPH_Util(object):

    # Type Annotation
    kissing_num: list[int]
    
    def __init__(self):
        
        self.kissing_num = [0, 2, 6, 12, 24]
        
    
    def init_particle_state(self, sph):
        
        # Variables
        n_dim = sph.sim_param.n_dim
        h = sph.particle_model.h
        
        # Compute No. of Particles
        sph.n_particle_G, n_per_dim = sph.sim_domain.compute_no_of_particles(h, n_dim)
        
        # Allocate share_memory space
        sph.mp_manager.assign_share_memory(sph.n_particle_G, sph.sim_param)

        # Array Shape
        shape_2D = sph.n_particle_G * n_dim
        shape_1D = sph.n_particle_G
        
        # Data Type
        dtype_float = sph.sim_param.float_precision.get_np_dtype()
        dtype_int = sph.sim_param.int_precision.get_np_dtype(signed=False)
        
        # Assign Position
        sph.x_G = np.ndarray(shape_2D, dtype = dtype_float, buffer=sph.mp_manager.shm['x_G'].buf)        
        self.recursion_build(sph, 0, [0 for i in range(n_dim)], n_per_dim, 0)
        
        # Assign 1D States
        sph.id_G = np.ndarray(shape_1D, dtype=dtype_int, buffer=sph.mp_manager.shm['id_G'].buf)
        sph.id_G = np.array([i for i in range(shape_1D)], dtype=dtype_int)

        sph.Ek_G = np.ndarray(shape_1D, dtype=dtype_float, buffer=sph.mp_manager.shm['Ek_G'].buf)
        sph.Ek_G = np.zeros(shape_1D, dtype=dtype_float)
        
        sph.Ep_G = np.ndarray(shape_1D, dtype=dtype_float, buffer=sph.mp_manager.shm['Ep_G'].buf)
        sph.Ep_G = np.zeros(shape_1D, dtype=dtype_float)
        
        sph.E_G = np.ndarray(shape_1D, dtype=dtype_float, buffer=sph.mp_manager.shm['E_G'].buf)
        sph.E_G = np.zeros(shape_1D, dtype=dtype_float)
        
        sph.rho_G = np.ndarray(shape_1D, dtype=dtype_float, buffer=sph.mp_manager.shm['rho_G'].buf)
        sph.rho_G = np.array([dtype_float(sph.particle_model.rho_0) for i in range(shape_1D)],
                             dtype=dtype_float)
        
        sph.p_G = np.ndarray(shape_1D, dtype=dtype_float, buffer=sph.mp_manager.shm['p_G'].buf)
        sph.p_G = np.zeros(shape_1D, dtype=dtype_float)
                
        # Assign 2D States
        sph.v_G = np.ndarray(shape_2D, dtype=dtype_float, buffer=sph.mp_manager.shm['v_G'].buf)
        sph.v_G = np.zeros(shape_2D, dtype=dtype_float)
        
        sph.a_G = np.ndarray(shape_2D, dtype=dtype_float, buffer=sph.mp_manager.shm['a_G'].buf)
        sph.a_G = np.zeros(shape_2D, dtype=dtype_float)

        # Assign Scalar
        sph.E_total_G = 0.0
        sph.Ek_total_G = 0.0
        sph.Ep_total_G = 0.0
        
    
    def recursion_build(self, sph: BaseSPH, loop_depth: int,
                        loop_index: list[int], loop_lim: list[int],
                        i_particle: int) -> int:
        
        while loop_index[loop_depth] < loop_lim[loop_depth]:
            
            # Loop Variables            
            n_dim = sph.sim_param.n_dim
            index = i_particle * n_dim
            init_pos = sph.sim_domain.init_pos

            if loop_depth == n_dim - 1: 
                
                for dim in range(n_dim):
                    
                    sph.x_G[index] = init_pos[dim][0] + sph.particle_model.h * loop_index[dim]
                    index += 1
                
                i_particle += 1
                    
                
            else:
                
                i_particle = self.recursion_build(sph, loop_depth+1, loop_index, loop_lim, i_particle)
                
                
            loop_index[loop_depth] += 1
        
        
        loop_index[loop_depth] = 0
        
        return i_particle
       
    
    @staticmethod
    def perturb_particle(sph: BaseSPH):
        
        # Define RNG Seed
        pid = os.getpid() # Process ID
        current_time = int(time.time()) # Current Time
        rng = np.random.default_rng(current_time % pid)
        
        # Apply Perturbation
        sph.x += 0.001 * rng.random(size=sph.x.shape) - 0.0005
        
    
    @staticmethod
    def plot(sph: BaseSPH, plt, ax):
        
        plt.cla()

        ax.plot(sph.x[0::2], sph.x[1::2], '.', markersize=10) 
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        
        plt.xlabel('$x$', fontsize=15, usetex=False)
        plt.ylabel('$y$', fontsize=15, usetex=False)
        plt.xlim(sph.sim_domain.bounding_box[0])
        plt.ylim(sph.sim_domain.bounding_box[1])
        
        plt.title('$t = {0:.3f} s$'.format(sph.sim_param.t), usetex=False)
        
        plt.show()
        
        if sph.sim_param.t == 0.0:
            plt.pause(1.0)
        else:
            plt.pause(0.05)
                        