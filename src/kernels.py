# Kernel

#%% Import

import numpy as np
import abc

#%% Main Class

class BaseKernel(object):

    n_dim: int
    h: float
    radius_of_influence: float
    alpha_dim: float
    
    def __init__(self, n_dim: int, smoothing_length: float, radius_of_influence: float):
        
        self.n_dim = n_dim
        self.h = smoothing_length
        self.radius_of_influence = radius_of_influence
        
        self.alpha_dim = float()
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
    
    def __init__(self, n_dim: int, smoothing_length: float, radius_of_influence: float):
        
        super().__init__(n_dim, smoothing_length, radius_of_influence)
        
    def _set_alpha_dim(self, n_dim: int):
        
        if n_dim == 2:
            self.alpha_dim = 10 / (7 * np.pi * np.power(self.h,2))
            
        elif n_dim == 3:
            self.alpha_dim = 21 / (16 * np.pi * np.power(self.h,3))          
            
            
    def W(self, q: np.ndarray) -> np.ndarray:
        
       W = self.alpha_dim * np.power((1.0 - 0.5 * q),4) * (2 * q + 1)
       
       return W
    
    
    def nabla_W(self, q: np.ndarray) -> np.ndarray:
        
        nabla_W = -self.alpha_dim / self.h * 0.625 * np.power((2 - q),3) * (q - 0.8) 
        
        return nabla_W


    def nabla2_W(self, q: np.ndarray) -> np.ndarray:
        
        q_m = q - 1.7
        h2 = np.power(self.h, 2)

        nabla2_W = self.alpha_dim / h2 * (2.5 * np.power(q_m, 3) - 0.675 * q_m + 0.135)
        
        return nabla2_W