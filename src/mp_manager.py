# MP Manager

#%% Import

import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

from simulations import SimulationParameter


#%% Main Class

class MP_Manager(object):
    
    # Type Annotation
    n_process: int
    shm: dict[str, shared_memory.SharedMemory]

    def __init__(self, n_process: int):
    
        # Process Attributes
        self.n_process = n_process
        self.shm = {}
        
        
    def assign_share_memory(self, n_particle_G: int, sim_param: SimulationParameter):
        
        # Sim Param
        n_dim = sim_param.n_dim
        
        #%% Float
        
        # Array Memory Size
        precision_byte = sim_param.float_precision.value

        # 2D Array
        mem_array_2D = int(n_particle_G * n_dim * precision_byte)
                
        self.shm['x_G'] = shared_memory.SharedMemory(create=True, name='x_G', size=mem_array_2D)
        self.shm['v_G'] = shared_memory.SharedMemory(create=True, name='v_G', size=mem_array_2D)
        self.shm['a_G'] = shared_memory.SharedMemory(create=True, name='a_G', size=mem_array_2D)
        
        # 1D Array
        mem_array_1D = n_particle_G * precision_byte
        
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

        mem_array_1D = int(n_particle_G * precision_byte)
        
        self.shm['id_G'] = shared_memory.SharedMemory(create=True, name='id_G',
                                                      size=mem_array_1D)
        
 
    def distribute_particle(self, data, data_G):
        pass
            
