# Adaptive Vector

#%% Import

import numpy as np
import numpy.typing as npt

#%% Main Class

class AdaptiveVector(object):

    n_dim: int
    n_particle: int
    shape: tuple[int]

    data: np.ndarray
    end_chunk_index: int
    gap_chunk_index: list[int]

    def __init__(self, n_dim: int, dtype: npt.DTypeLike):
        
        self.n_dim = n_dim
        self.n_particle = 0
        self.shape = tuple([self.n_dim * self.n_particle + 10])

        self.data = np.ndarray(shape=self.shape, dtype=dtype)
        self.end_chunk_index = 0
        self.gap_chunk_index = []


    def _resize_shape(self, exp_size: int):
        
        shape: tuple[int] = tuple([self.shape[0] + exp_size])
        self.data.resize(shape)


    def append_data(self, data: np.ndarray):

        size: int = data.shape[0] // self.n_dim

        start_index: int = self.n_particle * self.n_dim
        end_index: int = start_index + data.shape[0]

        if (self.shape[0] < end_index):
            self._resize_shape(end_index - self.shape[0] + 10)

        self.data[start_index: end_index] = data[:]

        self.n_particle += size
        self.end_chunk_index = self.n_particle

    
    def append_gap_chunk(self, index: int):

        self.gap_chunk_index.append(index)


    def fill_gap_chunk(self):

        self.gap_chunk_index.sort(reverse=True)

        if len(self.gap_chunk_index) == 0:
            return

        while self.end_chunk_index != self.n_particle:
            
            start_index: int = (self.end_chunk_index - 1) * self.n_dim
            end_index: int = start_index + self.n_dim

            gap_start_index: int = self.gap_chunk_index[-1] * self.n_dim
            gap_end_index: int = gap_start_index + self.n_dim

            self.data[gap_start_index:gap_end_index] = self.data[start_index:end_index]

            self.gap_chunk_index.pop()
            self.end_chunk_index -= 1

            self.gap_chunk_index = list(filter(lambda x: x <= self.end_chunk_index, self.gap_chunk_index))

        self.gap_chunk_index.clear()