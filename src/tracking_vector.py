# Dynamic Vector

#%% Import
import numpy as np

#%% Main Class

class TrackingVector(object):

    n_dim: int
    n: int
    shape: tuple[int]

    def __init__(self, n_dim: int, n: int):
        
        self.n_dim = n_dim
        self.n = n
        self.shape = tuple([self.n_dim * self.n])




if __name__ == "__main__":

    vec = TrackingVector(2,10)

