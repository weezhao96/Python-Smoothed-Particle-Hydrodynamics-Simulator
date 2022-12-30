# Simulation Run

#%% Library

from models import Atmosphere, Particle
from simulations import SimulationParameter, SimulationDomain
from kernels import QuinticKernel
from sph import BasicSPH, SPH_Util
from precision_enums import IntType, FloatType
from mp_manager import MP_Manager
from io_manager import IO_Manager

#%% Main

if __name__ == '__main__':

    #%% Model Definition
    
    # Atmosphere
    earth = Atmosphere(gravitational_strength=9.81)
    
    # Particle
    water = Particle(resting_density=1000.0, viscosity=1.0,
                     specific_heat_ratio=7.0, speed_of_sound=30.0)
    

    #%% Simulation Definition
    
    sim_param = SimulationParameter(n_dim=2, sim_duration=0.1, dt=0.001,
                                    float_precision=FloatType.FLOAT64, int_precision=IntType.INT32)
    
    unit_cube = SimulationDomain(bounding_box=[[0.0,1.0], [0.0,1.0]],
                                 initial_position=[[0.1,0.5], [0.1,0.2]])
    
    quintic = QuinticKernel(n_dim=sim_param.n_dim, smoothing_length=0.01, radius_of_influence=2.0)
    
    
    #%% Manager Definition
    
    mp_manager = MP_Manager(n_process=1)

    io_manager = IO_Manager(output_folder='output/')


    #%% SPH
    
    sph = BasicSPH(SPH_Util(), earth, water,
                   sim_param, unit_cube, quintic,
                   mp_manager, io_manager)
    
    sph.run_simulation()
    
    for attr in sph.__dict__:
                
        if attr[-2:] == '_G' and attr != 'n_particle_G' and attr[-7:] != 'total_G':
            print(' ')
            print('{}: {}'.format(attr[:-2], sph.__dict__[attr[:-2]].dtype))
            print('{}: {}'.format(attr, sph.__dict__[attr].dtype))
                                                     
        if attr == 'n_particle_G' or attr[-7:] == 'total_G':
            print(' ')
            print('{}: {}'.format(attr[:-2], type(sph.__dict__[attr[:-2]])))
            print('{}: {}'.format(attr, type(sph.__dict__[attr])))
