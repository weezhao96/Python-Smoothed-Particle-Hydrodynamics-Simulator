# MP Manager

#%% Import

import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

from precision_enum import IntType, FloatType

#%% Main Class

class MP_Manager(object):
    
    def __init__(self, n_process):
    
        # Process Attributes
        self.n_process = n_process
        self.shm = {}
        
    def assign_share_memory(self, n_particle_G, sim_param):
        
        # Sim Param
        n_dim = sim_param.n_dim
        
        #%% Float
        
        # Array Memory Size
        precision_byte = sim_param.float_precision.value

        mem_array_1D = n_particle_G * precision_byte
        mem_array_2D = n_particle_G * n_dim * precision_byte
                
        # 2D Array
        self.shm['x_G'] = shared_memory.SharedMemory(create=True, name='x_G',
                                                     size=mem_array_2D)
        self.shm['v_G'] = shared_memory.SharedMemory(create=True, name='v_G',
                                                     size=mem_array_2D)
        self.shm['a_G'] = shared_memory.SharedMemory(create=True, name='a_G',
                                                     size=mem_array_2D)
        
        # 1D Array
        self.shm['rho_G'] = shared_memory.SharedMemory(create=True, name='rho_G',
                                                       size=mem_array_1D)
        self.shm['p_G'] = shared_memory.SharedMemory(create=True, name='p_G',
                                                     size=mem_array_1D)
        self.shm['Ek_G'] = shared_memory.SharedMemory(create=True, name='Ek_G',
                                                      size=mem_array_1D)
        self.shm['Ep_G'] = shared_memory.SharedMemory(create=True, name='Ep_G',
                                                      size=mem_array_1D)
        self.shm['E_G'] = shared_memory.SharedMemory(create=True, name='E_G',
                                                     size=mem_array_1D)   
        
        #%% Int
        
        # Array Memory Size
        precision_byte = sim_param.int_precision.value

        mem_array_1D = n_particle_G * precision_byte
        
        self.shm['id_G'] = shared_memory.SharedMemory(create=True, name='id_G',
                                                      size=mem_array_1D)
        
        
        