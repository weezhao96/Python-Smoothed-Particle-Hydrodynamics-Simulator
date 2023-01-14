# MP Manager

#%% Import

from __future__ import annotations
from typing import Callable, Iterable
from simulations import SimulationParameter, SimulationDomain
from multiprocessing import Process
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Barrier
from multiprocessing.connection import Connection

import multiprocessing as mp
import numpy as np
import random


#%% Main Class

class MP_Manager(object):
    
    # Type Annotation
    n_proc: int
    shm: dict[str, SharedMemory]

    current_proc: BaseProcess
    proc_id: int

    grid: Grid

    global_comm: GlobalComm
    local_comm: LocalComm


    def __init__(self, n_dim: int, n_process: int):

        # Process Attributes
        self.n_proc = n_process
        self.shm = {}

        # Grid
        self.grid = Grid(grid_n_dim=n_dim-1)

        # Comms Channel
        self.global_comm = GlobalComm(barrier=mp.Barrier(self.n_proc))
        self.local_comm = None

        # Child Process
        self.current_proc = None
        self.proc_id = None

    
    def setup_parent(self, sim_domain: SimulationDomain, sim_param: SimulationParameter, n_particle_G: int):

        self.grid.setup(sim_domain, self.n_proc)

        self._assign_share_memory(sim_param, n_particle_G)


    def _assign_share_memory(self, sim_param: SimulationParameter, n_particle_G: int):
        
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


    def init_and_start_processes(self, core_simulation: Callable):

        procs: list[Process] = []
        local_comms: list[LocalComm] = []

        local_comms = self._generate_local_comms()

        for id in range(self.n_proc):

            proc = Process(target=core_simulation, args=(local_comms[id],), name='{}'.format(id))
            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()


    def _generate_local_comms(self) -> list[LocalComm]:
        
        n_dim = self.grid.n_dim
        
        # Generate Vectors
        vectors: list[tuple[int, ...]] = []

        if n_dim == 1:

            for i in range(-1,2):

                vector = []
                vector.append(i)

                vectors.append(tuple(vector))

            in_comms : list[Connection] = []
            out_comms : list[Connection] = []

            for i in range(3):
                
                in_comms.append(None)
                out_comms.append(None)

        elif n_dim == 2:

            for i in range(-1,2):
                for j in range(-1,2):

                    vector = []
                    vector.append(i)
                    vector.append(j)

                    vectors.append(tuple(vector))

            in_comms : list[list[Connection]] = [[],[],[]]
            out_comms : list[list[Connection]] = [[],[],[]]

            for i in range(3):
                for j in range(3):

                    in_comms[i].append(None)
                    out_comms[i].append(None)


        # Generate LocalComm
        local_comms: list[LocalComm] = []

        for id in range(self.n_proc):

            local_comms.append(LocalComm(out_comms=out_comms, in_comms=in_comms))

        # Generate Pipes
        for id in range(self.n_proc):
            
            out_coord = self.grid.proc2grid[id]

            for vector in vectors:
            
                in_coord = np.array(out_coord) + np.array(vector)

                if tuple(in_coord) in self.grid.grid2proc:

                    id_in = self.grid.grid2proc[tuple(in_coord)]
                    
                    in_conn, out_conn = mp.Pipe()

                    if n_dim == 1:

                        local_comms[id].out_comms[vector[0]] = out_conn
                        local_comms[id_in].in_comms[-vector[0]] = in_conn

                    elif n_dim == 2:

                        local_comms[id].out_comms[vector[0]][vector[1]] = out_conn
                        local_comms[id_in].in_comms[-vector[0]][-vector[1]] = in_conn

        return local_comms


    def setup_children(self, local_comm: LocalComm):

        self._get_current_proc()
        self.local_comm = local_comm


    def _get_current_proc(self):

        self.current_proc = mp.current_process()
        self.proc_id = int(self.current_proc.name)


#%% Global Comm

class GlobalComm(object):

    barrier: Barrier

    def __init__(self, barrier: Barrier):

        self.barrier = barrier


    def sync_processes(self):

        self.barrier.wait()


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


#%% LocalComm

class LocalComm(object):

    def __init__(self, out_comms, in_comms):
        
        self.out_comms = out_comms
        self.in_comms = in_comms


#%% Grid

class Grid(object):

    n_dim: int

    grid_size: tuple[int, ...]
    grid_spacing: tuple[float, ...]

    grid2proc: dict[tuple[int, ...], int]
    proc2grid: dict[int, tuple[int, ...]]

    bounds: dict[int, list[list[float]]]

    def __init__(self, grid_n_dim: int) -> None:
        
        # Grid Variables
        self.n_dim = grid_n_dim

        self.grid_size = tuple()
        self.grid_spacing = tuple()

        self.grid2proc = {}
        self.proc2grid = {}

        self.bounds = {}


    def setup(self, sim_domain: SimulationDomain, n_proc: int):

        self._compute_grid_size(sim_domain, n_proc)
        self._assign_grid(n_proc)
        self._compute_bounds(n_proc)


    def _compute_grid_size(self, sim_domain: SimulationDomain, n_proc: int):

        if self.n_dim == 1:

            self.grid_size = tuple([n_proc])

        elif self.n_dim == 2:

            root = np.sqrt(n_proc)
            ceil = int(np.ceil(root))

            grid_size: list[int] = []
            prev_diff: float = float(n_proc * n_proc)

            for i in range(ceil,1,-1):

                j = n_proc // i

                if i * j == n_proc:

                    diff = np.abs(root - (i + j))

                    if diff < prev_diff:

                        prev_diff = diff
                        grid_size = [i,j]


            self.grid_size = tuple(grid_size)

        # Grid Spacing
        grid_spacing: list = []

        for dim in range(self.n_dim):

            distance = sim_domain.domain[dim][1] - sim_domain.domain[dim][0]

            grid_spacing.append(distance / self.grid_size[dim])

        self.grid_spacing = tuple(grid_spacing)


    def _assign_grid(self, n_proc: int):

        # Mappings
        # -- Grid Dimension == 1
        if self.n_dim == 1:

            for proc in range(n_proc):

                self.proc2grid[proc] = tuple([proc])
                self.grid2proc[tuple([proc])] = proc

        # -- Grid Dimension == 2
        if self.n_dim == 2:

            proc = 0

            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):

                    tuple_ij = tuple([i,j])

                    self.proc2grid[proc] = tuple_ij
                    self.grid2proc[tuple_ij] = proc

                    proc += 1


    def _compute_bounds(self, n_proc: int):

        # Compute Bound
        for proc in range(n_proc):
            
            bound: list[list[float]] = []

            grid = self.proc2grid[proc]
            
            for dim in range(self.n_dim):

                lwr_bound = self.grid_spacing[dim] * grid[dim]
                upr_bound = self.grid_spacing[dim] * (grid[dim] + 1)

                bound.append([lwr_bound, upr_bound])
            
            self.bounds[proc] = bound


