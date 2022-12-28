# IO Manager

#%% Import

from __future__ import annotations

from threading import Thread
from queue import PriorityQueue

import abc
import os
import shutil
import numpy as np

#%% Main Class

class IO_Manager(object):
    
    # Type Annotation
    output_folder: str
    state_writer: StateWriter
    energy_writer: EnergyWriter

    def __init__(self, output_folder: str):
    
        # Output Folder
        self.output_folder = output_folder

        self.construct_output_dir()

        # Writer
        self.state_writer = StateWriter(self.output_folder + '/states')
        self.energy_writer = EnergyWriter(self.output_folder)


    def construct_output_dir(self):

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        os.makedirs(self.output_folder + '/states')


class WriterService(object):

    def __init__(self, output_folder: str):
        
        # Output Folder
        self.output_folder = output_folder

        # Queue
        self.queue = PriorityQueue()

        # Threads
        self.writer = None
        self.parsers = None

    def create_writer_thread(self, filename):

        self.writer = Thread(target=self._write, args=(filename, self.queue), daemon=True)
        self.writer.start()


    @staticmethod        
    def _write(filename: str, queue: PriorityQueue):

        with open(filename, 'w+') as file:

            while True:

                priority, line = queue.get()

                if line == None:
                    break

                else:

                    file.write(line)
                    file.flush()

                    queue.task_done()

        queue.task_done()


    def sync_queue(self, n_particle_G: int):

        self.queue.put((n_particle_G,None))
        self.queue.join()


    @abc.abstractmethod
    def output_data(self):
        pass

    @abc.abstractstaticmethod
    def _parse():
        pass


class StateWriter(WriterService):

    def output_data(self, n_particle_G: int, n_dim: int, t: float, t_count: int,
                    x_G: np.ndarray, v_G: np.ndarray, a_G: np.ndarray,
                    rho_G: np.ndarray, p_G: np.ndarray):

        self.writer = None
        self.parsers = None

        # Writer Thread Creation
        filename = self.output_folder + '/t_{:0>5}_states.txt'.format(t_count)

        self.create_writer_thread(filename)

        # Parse Time and Header
        line = 't = {0} \n'.format(t)
        self.queue.put((-2,line))

        line = '{0:>20} {1:>20} {2:>20} '.format('n','rho','p')

        for dim in range(n_dim):
            line = line + '{0:>20} '.format('x_{0}'.format(dim))

        for dim in range(n_dim):
            line = line + '{0:>20} '.format('v_{0}'.format(dim))    

        for dim in range(n_dim):
            line = line + '{0:>20} '.format('a_{0}'.format(dim))

        self.queue.put((-1,line + '\n'))
        
        # Parse Data
        self.parsers = []

        index_2D = 0

        for i in range(n_particle_G):

            args = (n_dim, i, rho_G[i], p_G[i], x_G[index_2D : index_2D + n_dim],
                    v_G[index_2D : index_2D + n_dim], a_G[index_2D : index_2D + n_dim],
                    self.queue)

            self.parsers.append(Thread(target=self._parse, args=args))

            index_2D += n_dim


        for thread in self.parsers:
            thread.start()

        for thread in self.parsers:
            thread.join()


    @staticmethod
    def _parse(n_dim: int, id: int, rho: np.float_, p: np.float_,
               x: np.ndarray, v: np.ndarray, a: np.ndarray, queue: PriorityQueue):

        # Output Data
        line = '{0:>20} {1:>20} {2:>20} '.format(id, rho, p)

        for dim in range(n_dim):
            line = line + '{0:>20} '.format(x[dim])

        for dim in range(n_dim):
            line = line + '{0:>20} '.format(v[dim])

        for dim in range(n_dim):
            line = line + '{0:>20} '.format(a[dim])

        # Insert to Queue
        queue.put((id,line + '\n'))


class EnergyWriter(WriterService):

    def __init__(self, output_folder: str):

        super().__init__(output_folder=output_folder)

        # Writer Thread Creation
        filename = self.output_folder + 'energy.txt'
        self.create_writer_thread(filename)

        # Parse Header
        line = '{0:>20} {1:>20} {2:>20} {3:>20} \n'.format('t', 'E_kinetic', 'E_potential', 'E_total')
        self.queue.put((-1,line))


    def output_data(self, t: float, Ek_total: np.float_, Ep_total: np.float_, E_total: np.float_):
        
        self._parse(t, Ek_total, Ep_total, E_total, self.queue)


    @staticmethod
    def _parse(t: float, Ek_total: np.float_, Ep_total: np.float_, E_total: np.float_, queue: PriorityQueue):

        line = '{0:>20} {1:>20} {2:>20} {3:>20} \n'.format(t, Ek_total, Ep_total, E_total)
        queue.put((0,line))