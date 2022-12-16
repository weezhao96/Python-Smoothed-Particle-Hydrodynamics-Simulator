# Kernel

#%% Import

import numpy as np

#%% Main Class

class BaseKernel(object):
    
    def __init__(self, n_dim, radius_of_influence):
        
        self.n_dim = n_dim
        self.h = radius_of_influence
        
        self.alpha_dim = self._set_alpha_dim(n_dim)


    def _set_alpha_dim(n_dim):
        
        pass


    def W(q):
        
        pass


    def nable_W(q):
        
        pass
    
    
#%% Quintic

class QuinticKernel(BaseKernel):
    
    def __init__(self, n_dim, radius_of_influence):
        
        super().__init__(n_dim, radius_of_influence)
        
        
    def _set_alpha_dim(self, n_dim):
        
        if n_dim == 2:
            self.alpha_dim = 10 / (7 * np.pi * np.power(self.h,2))
            
        elif n_dim == 3:
            self.alpha_dim = 21 / (16 * np.pi * np.power(self.h,3))
            
            
    def W(self, q):
        
       W = self.alpha_dim * np.power((1.0 - 0.5 * q),4) * (2 * q + 1)
       
       return W
   
    
    def nabla_W(self, q):
        
        q4 = np.power(q,4)
        q3 = np.power(q,3)
        p2 = np.power(q,2)
        
        nabla_W = self.alpha_dim / self.h * (0.625 * q4 - 3.75 * q3 + 7.5 * q2 - 5 * q)
        
        return nabla_W
        