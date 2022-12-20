# SPH Utilities

#%% Import

import numpy as np

#%% Class Definition

class SPH_Util(object):
    
    def __init__(self):
        pass
    
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
        dtype_int = sph.sim_param.int_precision.get_np_dtype()
        
        #%% Assign Position
        
        # Position
        sph.x_G = np.ndarray(shape_2D, dtype = dtype_float, buffer=sph.mp_manager.shm['x_G'].buf)
        
        self.recursion_build(sph, 0, [0 for i in range(n_dim)], n_per_dim, 0)
        
        #%% Assign 1D and 2D States
        
        sph.id_G = np.ndarray(shape_1D, dtype = dtype_int, buffer=sph.mp_manager.shm['id_G'].buf)
        sph.id_G = np.array([i for i in range(shape_1D)])
        
        sph.v_G = np.ndarray(shape_2D, dtype = dtype_float, buffer=sph.mp_manager.shm['v_G'].buf)
        sph.v_G = np.array([0.0 for i in range(shape_2D)]) 
        
        sph.a_G = np.ndarray(shape_2D, dtype = dtype_float, buffer=sph.mp_manager.shm['a_G'].buf)
        sph.a_G = np.array([0.0 for i in range(shape_2D)])
        
        sph.rho_G = np.ndarray(shape_1D, dtype = dtype_float, buffer=sph.mp_manager.shm['rho_G'].buf)
        sph.rho_G = np.array([sph.particle_model.rho_0 for i in range(shape_1D)])
        
        sph.p_G = np.ndarray(shape_1D, dtype = dtype_float, buffer=sph.mp_manager.shm['p_G'].buf)
        sph.p_G = np.array([1000.0 for i in range(shape_1D)])
        
        
    @staticmethod
    def perturb_particle(sph):
        pass
    

    def recursion_build(self, sph, loop_depth, loop_index, loop_lim, i_particle):
        
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
                
                i_particle = self.recursion_build(sph, loop_depth + 1, loop_index, loop_lim,
                                                  i_particle)
                
                
            loop_index[loop_depth] += 1
        
        
        loop_index[loop_depth] = 0
        
        return i_particle
            
            
    @staticmethod
    def plot(sph, plt, ax):
        
        plt.cla()
        print(sph.x_G)
        ax.plot(sph.x_G[::2], sph.x_G[1::2], '.', markersize=10) 
        ax.grid(True)
        ax.set_aspect('equal', 'box')
        
        plt.xlabel('$x$', fontsize=15, usetex=True)
        plt.ylabel('$y$', fontsize=15, usetex=True)
        plt.xlim(sph.sim_domain.bounding_box[0])
        plt.ylim(sph.sim_domain.bounding_box[1])
        
        plt.title('$t = {0} \, s$'.format(sph.sim_param.t), usetex=True)
        
        plt.show()
        plt.pause(0.05)