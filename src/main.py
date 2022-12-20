# Simulation Run

#%% Library

import numpy as np

from models import Atmosphere, Particle
from simulations import SimulationParameter, SimulationDomain
from kernels import QuinticKernel
from sph import BasicSPH
from precision_enums import IntType, FloatType
from sph_util import SPH_Util


#%% Main

if __name__ == '__main__':

    #%% Simulation Precision
    
    float_precision = FloatType.FLOAT32
    float_dtype = float_precision.get_np_dtype()
    
    int_precision = IntType.INT16
    int_dtype = int_precision.get_np_dtype()

    #%% Model Definition
    
    # Atmosphere
    earth = Atmosphere(gravitational_strength=float_dtype(9.81))
    
    # Particle
    water = Particle(radius_of_influence=float_dtype(0.01),
                     resting_density=float_dtype(1000.0), viscosity=float_dtype(1.0),
                     specific_heat_ratio=float_dtype(7.0), speed_of_sound=float_dtype(1480.0))
    
    #%% Simulation Definition
    
    sim_param = SimulationParameter(n_dim=int_dtype(2),
                                    sim_duration=float_dtype(5.0), dt=float_dtype(0.1),
                                    float_precision=float_precision, int_precision=int_precision)
    
    unit_cube = SimulationDomain(bounding_box=np.array([[0.0,1.0],
                                                        [0.0,1.0]], dtype=float_dtype),
                                 initial_position=np.array([[0.5,0.6],
                                                            [0.5,0.6]], dtype=float_dtype))
    
    quintiq = QuinticKernel(n_dim=sim_param.n_dim, radius_of_influence=water.h)
    
    #%% SPH
    
    sph = BasicSPH(SPH_Util(), earth, water,
                   sim_param, unit_cube, quintiq,
                   1, "output/")
    
    sph.run_simulation()
    
    sph.clean_up_simulation()