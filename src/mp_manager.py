# MP Manager

#%% Import

import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

#%% Main Class

class MP_Manager(object):
    
    def __init__(self, n_process):
    
        # Process Attributes
        self.n_process = n_process
        self.shm = None
        
    def assign_share_memory(self, n_particle_G, n_dim):
        
        # Compute Total Memory
        mem_array_1D = n_particle_G * 8
        mem_array_2D = n_particle_G * n_dim * 8
                
        self.shm = shared_memory.SharedMemory(create=True, name='x_G',
                                              size=mem_array_2D)