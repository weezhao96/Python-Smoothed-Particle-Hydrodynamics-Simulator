# Interaction Set

#%% Import

import numpy as np

#%% Main Definition

class interaction_set(object):

    # Type Annotation
    n_dim: int
    id: int
    n_neighbour: int
    id_neighbour: np.ndarray
    q: np.ndarray
    dr: np.ndarray
    dv: np.ndarray

    def __init__(self, id: int, kissing_number: int, n_dim: int,
                 float_dtype: np.float_, int_dtype: np.int_):
        
        self.n_dim = n_dim

        self.id = id
        self.n_neighbour = int()

        self.id_neighbour = np.ndarray(shape=kissing_number, dtype=int_dtype)
        self.q = np.ndarray(shape=kissing_number, dtype=float_dtype)
        self.dr = np.ndarray(shape=kissing_number*n_dim, dtype=float_dtype)
        self.dv = np.ndarray(shape=kissing_number*n_dim, dtype=float_dtype)

        self.index_1D = 0
        self.index_2D = 0


    def add_neighbour(self, id_neighbour: int, q: np.ndarray, dr: np.ndarray, dv: np.ndarray):

        self.n_neighbour += 1
        
        self.id_neighbour[self.index_1D] = id_neighbour
        self.q[self.index_1D] = q
        self.index_1D += 1

        self.dr[self.index_2D : self.index_2D + self.n_dim] = dr
        self.dv[self.index_2D : self.index_2D + self.n_dim] = dv
        self.index_2D += self.n_dim


    def loop_reset(self):

        self.n_neighbour = 0
        self.index_1D = 0
        self.index_2D = 0