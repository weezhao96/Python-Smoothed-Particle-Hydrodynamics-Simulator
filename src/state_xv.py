# State XV

#%% Import

from dataclasses import dataclass
import numpy as np

#%% Main Class

@dataclass
class StateXV(object):

    id: int
    x: np.float_
    v: np.float_