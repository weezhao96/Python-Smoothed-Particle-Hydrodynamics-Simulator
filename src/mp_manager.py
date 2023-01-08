# MP Manager

#%% Import

from __future__ import annotations
from simulations import SimulationParameter, SimulationDomain
from typing import TYPE_CHECKING, Optional
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

if TYPE_CHECKING:
    from sph import BaseSPH, BasicSPH

import numpy as np
import random


#%% Main Class

class MP_Manager(object):
    
    # Type Annotation
    n_proc: int
    shm: dict[str, SharedMemory]

    grid_n_dim: int

    grid_size: tuple[int, ...]
    grid_spacing: tuple[float, ...]

    grid2proc: dict[tuple[int, ...], int]
    proc2grid: dict[int, tuple[int, ...]]

    bounds: dict[int, list[list[float]]]

    def __init__(self, n_dim: int, n_process: int):
    
        # Process Attributes
        self.n_proc = n_process
        self.shm = {}

        # Grid Variables
        self.grid_n_dim = n_dim - 1

        self.grid_size = tuple()
        self.grid_spacing = tuple()

        self.grid2proc = {}
        self.proc2grid = {}

        self.bounds = {}

        def is_prime(n: int) -> bool:

            """
            Miller-Rabin primality test.

            A return value of False means n is certainly not prime. A return value of
            True means n is very likely a prime.
            """

            n = int(n)

            # Miller-Rabin test for prime

            if n == 0 or n == 1 or n == 4 or n == 6 or n == 8 or n == 9:
                return False
                
            if n == 2 or n == 3 or n == 5 or n == 7:
                return True

            s = 0
            d = n-1

            while d % 2 == 0:
                d >>= 1
                s += 1
            assert(2**s * d == n-1)
        
            def trial_composite(a):

                if pow(a, d, n) == 1:
                    return False

                for i in range(s):

                    if pow(a, 2**i * d, n) == n-1:
                        return False

                return True  
        
            for i in range(8):# No. of Trials
                a = random.randrange(2, n)

                if trial_composite(a):
                    return False
        
            return True

        # Setup
        if self.grid_n_dim == 2 and is_prime(self.n_proc):
            self.n_proc -= 1

    
    def setup(self, sim_domain: SimulationDomain, sim_param: SimulationParameter, n_particle_G: int):

        self._compute_grid_size(sim_domain)
        self._assign_grid()

        self._compute_bounds()

        self._assign_share_memory(n_particle_G, sim_param)

    
    def _compute_grid_size(self, sim_domain: SimulationDomain):

        if self.grid_n_dim == 1:

            self.grid_size = tuple([self.n_proc])

        elif self.grid_n_dim == 2:

            root = np.sqrt(self.n_proc)
            ceil = int(np.ceil(root))

            grid_size: list[int] = []
            prev_diff: float = float(self.n_proc * self.n_proc)

            for i in range(ceil,1,-1):

                j = self.n_proc // i

                if i * j == self.n_proc:

                    diff = np.abs(root - (i + j))

                    if diff < prev_diff:

                        prev_diff = diff
                        grid_size = [i,j]


            self.grid_size = tuple(grid_size)

        # Grid Spacing
        grid_spacing: list = []

        for dim in range(self.grid_n_dim):

            distance = sim_domain.domain[dim][1] - sim_domain.domain[dim][0]

            grid_spacing.append(distance / self.grid_size[dim])

        self.grid_spacing = tuple(grid_spacing)


    def _assign_grid(self):

        # Mappings
        # -- Grid Dimension == 1
        if self.grid_n_dim == 1:

            for proc in range(self.n_proc):

                self.proc2grid[proc] = tuple([proc])
                self.grid2proc[tuple([proc])] = proc

        # -- Grid Dimension == 2
        if self.grid_n_dim == 2:

            proc = 0

            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):

                    tuple_ij = tuple([i,j])

                    self.proc2grid[proc] = tuple_ij
                    self.grid2proc[tuple_ij] = proc

                    proc += 1


    def _compute_bounds(self):

        # Compute Bound
        for proc in range(self.n_proc):
            
            bound: list[list[float]] = []

            grid = self.proc2grid[proc]
            
            for dim in range(self.grid_n_dim):

                lwr_bound = self.grid_spacing[dim] * grid[dim]
                upr_bound = self.grid_spacing[dim] * (grid[dim] + 1)

                bound.append([lwr_bound, upr_bound])
            
            self.bounds[proc] = bound


    def _assign_share_memory(self, n_particle_G: int, sim_param: SimulationParameter):
        
        # Sim Param
        n_dim = sim_param.n_dim
                
        # Array Memory Size
        # -- Float
        precision_byte = sim_param.float_prec.value

        # 2D Array
        mem_array_2D = n_particle_G * n_dim * precision_byte
                
        self.shm['x_G'] = SharedMemory(create=True, name='x_G', size=mem_array_2D)
        self.shm['v_G'] = SharedMemory(create=True, name='v_G', size=mem_array_2D)
        self.shm['a_G'] = SharedMemory(create=True, name='a_G', size=mem_array_2D)
        
        # 1D Array
        mem_array_1D = n_particle_G * precision_byte
        
        self.shm['rho_G'] = SharedMemory(create=True, name='rho_G', size=mem_array_1D)                               
        self.shm['p_G'] = SharedMemory(create=True, name='p_G', size=mem_array_1D)
        self.shm['Ek_G'] = SharedMemory(create=True, name='Ek_G', size=mem_array_1D)
        self.shm['Ep_G'] = SharedMemory(create=True, name='Ep_G', size=mem_array_1D)
        self.shm['E_G'] = SharedMemory(create=True, name='E_G', size=mem_array_1D)
        
        # -- Int
        precision_byte = sim_param.int_prec.value

        mem_array_1D = int(n_particle_G * precision_byte)
        
        self.shm['id_G'] = SharedMemory(create=True, name='id_G', size=mem_array_1D)


    def comm_G2L(self, n_dim: int, n_particle: int, id: np.ndarray,
                 array: np.ndarray, array_G: np.ndarray, nd_array: int):
                
        if nd_array == 1:

            for i in range(n_particle):

                array[i] = array_G[id[i]]

        elif nd_array == 2:
            
            index = 0

            for i in range(n_particle):

                index_G = id[i] * n_dim

                array[index : index + n_dim] = array_G[index_G : index_G + n_dim]

                index += n_dim


    def comm_L2G(self, n_dim: int, n_particle: int, id: np.ndarray,
                 array: np.ndarray, array_G: np.ndarray, nd_array: int):
        
        if nd_array == 1:

            for i in range(n_particle):

                array_G[id[i]] = array[i]

        elif nd_array == 2:
            
            index = 0

            for i in range(n_particle):

                index_G = id[i] * n_dim

                array_G[index_G : index_G + n_dim] = array[index : index + n_dim]

                index += n_dim
                