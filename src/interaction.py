# Interaction Set

#%% Import

from simulations import SimulationParameter

import numpy as np


#%% Main Definition

class Interaction(object):

    # Type Annotation
    n_dim: int
    id: int
    n_neighbour: int
    id_neighbour: np.ndarray
    q: np.ndarray
    dr: np.ndarray
    dv: np.ndarray

    def __init__(self, id: int, sim_param: SimulationParameter):
        
        kissing_number: list[int] = [0, 2, 6, 12, 24]

        # Helper Variables
        self.n_dim = sim_param.n_dim

        shape_1D = int(1.5 * kissing_number[self.n_dim])
        shape_2D = shape_1D * self.n_dim

        float_dtype = sim_param.float_prec.get_np_dtype()
        int_dtype = sim_param.int_prec.get_np_dtype(signed=False)

        # Allocation
        self.id = id
        self.n_neighbour = int()

        self.id_neighbour = np.ndarray(shape=shape_1D, dtype=int_dtype)
        self.q = np.ndarray(shape=shape_1D, dtype=float_dtype)
        self.dr = np.ndarray(shape=shape_2D, dtype=float_dtype)
        self.dv = np.ndarray(shape=shape_2D, dtype=float_dtype)

        self.index_1D = 0
        self.index_2D = 0


    def add_neighbour(self, id: int, q: float, dr: np.ndarray, dv: np.ndarray):

        self.n_neighbour += 1
        
        self.id_neighbour[self.index_1D] = id
        self.q[self.index_1D] = q
        self.index_1D += 1

        self.dr[self.index_2D : self.index_2D + self.n_dim] = dr
        self.dv[self.index_2D : self.index_2D + self.n_dim] = dv
        self.index_2D += self.n_dim


    def reset(self):

        self.n_neighbour = 0
        self.index_1D = 0
        self.index_2D = 0