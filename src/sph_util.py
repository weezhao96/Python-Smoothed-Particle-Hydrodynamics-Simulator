# SPH Utilities

#%% Import

import numpy as np
import matplotlib.pyplot as plt

#%% Class Definition

class SPH_Util(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def init_particle_state(sph):
        
        # Array Shape
        shape_2D = sph.n_particle_G * sph.sim_param.n_dim
        shape_1D = sph.n_particle_G
        
        # Data Type
        dtype_float = sph.sim_param.float_precision.get_np_dtype()
        dtype_int = sph.sim_param.int_precision.get_np_dtype()
        
        # State Allocation
        # sph.x_G = np.ndarray(shape_2D, dtype = dtype_float,
        #                      buffer=sph.mp_manager.shm['x_G'].buf)
        # sph.x_G = np.array([0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5,
        #                     0.5, 0.5, 0.6, 0.5, 0.7, 0.5, 0.8, 0.5, 0.9, 0.5])
        
        # sph.v_G = np.ndarray(shape_2D, dtype = dtype_float,
        #                      buffer=sph.mp_manager.shm['v_G'].buf)
        # sph.v_G = np.array([0.0 for i in range(shape_2D)]) 
        
        # sph.a_G = np.ndarray(shape_2D, dtype = dtype_float,
        #                      buffer=sph.mp_manager.shm['a_G'].buf)
        # sph.a_G = np.array([0.0 for i in range(shape_2D)])
        
        # sph.id_G = np.ndarray(shape_1D, dtype = dtype_int,
        #                       buffer=sph.mp_manager.shm['id_G'].buf)
        # sph.id_G = np.array([i for i in range(shape_1D)])
        
        # sph.rho_G = np.ndarray(shape_1D, dtype = dtype_float,
        #                        buffer=sph.mp_manager.shm['rho_G'].buf)
        # sph.rho_G = np.array([1000.0 for i in range(shape_1D)])
        
        # sph.p_G = np.ndarray(shape_1D, dtype = dtype_float,
        #                      buffer=sph.mp_manager.shm['p_G'].buf)
        # sph.p_G = np.array([1000.0 for i in range(shape_1D)])
        
        sph.x = np.ndarray(shape_2D, dtype = dtype_float,
                             buffer=sph.mp_manager.shm['x_G'].buf)
        sph.x = np.array([0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4, 0.5,
                            0.5, 0.5, 0.6, 0.5, 0.7, 0.5, 0.8, 0.5, 0.9, 0.5])
        
        sph.v = np.ndarray(shape_2D, dtype = dtype_float,
                             buffer=sph.mp_manager.shm['v_G'].buf)
        sph.v = np.array([0.0 for i in range(shape_2D)]) 
        
        sph.a = np.ndarray(shape_2D, dtype = dtype_float,
                             buffer=sph.mp_manager.shm['a_G'].buf)
        sph.a = np.array([0.0 for i in range(shape_2D)])
        
        sph.id = np.ndarray(shape_1D, dtype = dtype_int,
                              buffer=sph.mp_manager.shm['id_G'].buf)
        sph.id = np.array([i for i in range(shape_1D)])
        
        sph.rho = np.ndarray(shape_1D, dtype = dtype_float,
                               buffer=sph.mp_manager.shm['rho_G'].buf)
        sph.rho = np.array([1000.0 for i in range(shape_1D)])
        
        sph.p = np.ndarray(shape_1D, dtype = dtype_float,
                             buffer=sph.mp_manager.shm['p_G'].buf)
        sph.p = np.array([1000.0 for i in range(shape_1D)])
        
    @staticmethod
    def perturb_particle(sph):
        pass
    
    @staticmethod
    def plot(sph, plt, ax):
        
        plt.cla()
        
        ax.plot(sph.x[::2], sph.x[1::2], '.', markersize=15) 
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        
        plt.xlabel('$x$', fontsize=15, usetex=True)
        plt.ylabel('$y$', fontsize=15, usetex=True)
        plt.xlim(sph.sim_domain.bounding_box[0])
        plt.ylim(sph.sim_domain.bounding_box[1])
        
        plt.title('$t = {0} s$'.format(sph.sim_param.t), usetex=True)
        
        plt.show()
        plt.pause(0.05)
