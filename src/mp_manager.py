#%% MP Manager

#%% Import

import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

#%% Main Class

class MP_Manager(object):
    
    def __init__(self, n_process):
    
        # Process Attributes
        self.n_process = n_process
        