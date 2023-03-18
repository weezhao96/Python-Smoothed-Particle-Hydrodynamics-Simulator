# Dynamic Vector

#%% Import
from typing import Union
import numpy as np
import numpy.typing as npt

#%% Main Class

class AdaptiveVector(object):

    n_dim: int
    n_particle: int
    shape: tuple[int]

    data: np.ndarray
    tail_chunk_index: int
    gap_chunk_index: list[int]

    def __init__(self, n_dim: int, n_particle: int, dtype: npt.DTypeLike):
        
        self.n_dim = n_dim
        self.n_particle = n_particle
        self.shape = tuple([self.n_dim * self.n_particle])


        self.data = np.ndarray(shape=self.shape, dtype=dtype)
        self.size = self.shape[0]
        self.gap_chunk_index = []
        self.


    def 


if __name__ == "__main__":

    vec = AdaptiveVector(2,10, np.int32)

    print(vec.shape)

