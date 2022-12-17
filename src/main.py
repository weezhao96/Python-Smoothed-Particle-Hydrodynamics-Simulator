# Simulation Run

#%% Library

from model import Atmosphere, Particle
from simulation import SimulationParameter, SimulationDomain
from kernel import QuinticKernel
from sph import BasicSPH
from precision_enum import IntType, FloatType
from sph_util import SPH_Util


#%% Main

if __name__ == '__main__':

    #%% Model Definition
    
    # Atmosphere
    earth = Atmosphere(gravitational_strength=9.81)
    
    # Particle
    water = Particle(radius_of_influence=0.01,
                     resting_density=1000.0, viscosity=1.0,
                     specific_heat_ratio=7.0, speed_of_sound=1480.0)
    
    #%% Simulation Definition
    
    sim_param = SimulationParameter(n_dim=2, sim_duration=1.0, dt=0.1,
                                    float_precision=FloatType.FLOAT32,
                                    int_precision=IntType.INT16)
    
    unit_cube = SimulationDomain(bounding_box=[[0.0,1.0],
                                               [0.0,1.0]],
                                 initial_position=[[0.25,0.75],
                                                   [0.25,0.75]])
    
    
    quintiq = QuinticKernel(n_dim=sim_param.n_dim, radius_of_influence=water.h)
    
    #%% SPH
    
    model = BasicSPH(SPH_Util(), earth, water,
                     sim_param, unit_cube, quintiq,
                     1, "output/")
    
    
    model.run_simulation()
    
    
    
    
    