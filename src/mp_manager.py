# MP Manager

#%% Import

import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

#%% Main Class

class MP_Manager(object):
    
    def __init__(self, n_process):
    
        # Process Attributes
        self.n_process = n_process
        self.shm = {}
        
    def assign_share_memory(self, n_particle_G, n_dim):
        
        # Array Memory Size
        mem_array_1D = n_particle_G * 8
        mem_array_2D = n_particle_G * n_dim * 8
                
        # 2D Array
        self.shm['x_G'] = shared_memory.SharedMemory(create=True, name='x_G',
                                                     size=mem_array_2D)
        self.shm['v_G'] = shared_memory.SharedMemory(create=True, name='v_G',
                                                     size=mem_array_2D)
        self.shm['a_G'] = shared_memory.SharedMemory(create=True, name='a_G',
                                                     size=mem_array_2D)
        
        # 1D Array
        self.shm['id_G'] = shared_memory.SharedMemory(create=True, name='id_G',
                                                      size=mem_array_1D)
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