# Simulation Run

#%% Library

from models import Atmosphere, Particle
from simulations import SimulationParameter, SimulationDomain
from kernels import QuinticKernel
from sph import BasicSPH
from precision_enums import IntType, FloatType
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
    
    sim_param = SimulationParameter(n_dim=2, sim_duration=5.0, dt=0.1,
                                    float_precision=FloatType.FLOAT64, int_precision=IntType.INT16)
    
    unit_cube = SimulationDomain(bounding_box=[[0.0,1.0], [0.0,1.0]],
                                 initial_position=[[0.5,0.6], [0.5,0.6]])
    
    quintiq = QuinticKernel(n_dim=sim_param.n_dim, radius_of_influence=water.h)
    
    #%% SPH
    
    sph = BasicSPH(SPH_Util(), earth, water,
                   sim_param, unit_cube, quintiq,
                   1, "output/")
    
    sph.run_simulation()
    
    sph.clean_up_simulation()