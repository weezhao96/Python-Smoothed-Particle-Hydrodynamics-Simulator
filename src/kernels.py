# Kernel

#%% Import

import numpy as np
import abc

#%% Main Class

class BaseKernel(object):

    n_dim: int
    h: float
    alpha_dim: float
    
    def __init__(self, n_dim: int, radius_of_influence: float):
        
        self.n_dim = n_dim
        self.h = radius_of_influence
        
        self.alpha_dim = None
        self._set_alpha_dim(n_dim)


    @abc.abstractmethod
    def _set_alpha_dim(self, n_dim: int):
        pass

    @abc.abstractmethod
    def W(self, q: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def nabla_W(self, q: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def nabla2_W(self, q: np.ndarray) -> np.ndarray:
        pass
    
    
#%% Quintic

class QuinticKernel(BaseKernel):
    
    def __init__(self, n_dim: int, radius_of_influence: float):
        
        super().__init__(n_dim, radius_of_influence)
        
    def _set_alpha_dim(self, n_dim: int):
        
        if n_dim == 2:
            self.alpha_dim = 10 / (7 * np.pi * np.power(self.h,2))
            
        elif n_dim == 3:
            self.alpha_dim = 21 / (16 * np.pi * np.power(self.h,3))            
            
            
    def W(self, q: np.ndarray) -> np.ndarray:
        
       W = self.alpha_dim * np.power((1.0 - 0.5 * q),4) * (2 * q + 1)
       
       return W
    
    
    def nabla_W(self, q: np.ndarray) -> np.ndarray:
        
        q4 = np.power(q,4)
        q3 = np.power(q,3)
        q2 = np.power(q,2)
        
        nabla_W = self.alpha_dim / self.h * (0.625 * q4 - 3.75 * q3 + 7.5 * q2 - 5 * q)
        
        return nabla_W

    def nabla2_W(self, q: np.ndarray) -> np.ndarray:
        
        q3 = np.power(q,3)
        q2 = np.power(q,2)
        h2 = np.power(self.h, 2)

        nabla2_W = self.alpha_dim / h2 * (2.5 * q3 - 11.25 * q2 + 15.0 * q)
        
        return nabla2_W