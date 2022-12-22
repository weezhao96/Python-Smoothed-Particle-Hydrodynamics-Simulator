# Data Type Enumeration

#%% Import

import numpy as np

from enum import Enum, unique


#%% Integer Type

@unique
class IntType(Enum):
    
    # IntType.value: no. of bytes
    INT8 = 1
    INT16 = 2
    INT32 = 4
    INT64 = 8
    
    def get_np_dtype(self, signed: bool = True):
        
        if signed:
            
            if self.value == 1:
                return np.int8
            
            elif self.value == 2:
                return np.int16
            
            elif self.value == 4:
                return np.int32
            
            elif self.value == 8:
                return np.int64
            
            
        else:
            
            if self.value == 1:
                return np.uint8
            
            elif self.value == 2:
                return np.uint16
            
            elif self.value == 4:
                return np.uint32
            
            elif self.value == 8:
                return np.uint64 
    

#%% Float Type

@unique
class FloatType(Enum):
    
    # FloatType.value: no. of bytes
    FLOAT16 = 2
    FLOAT32 = 4
    FLOAT64 = 8
    
    def get_np_dtype(self):
        
        if self.value == 2:
            return np.float16
        
        elif self.value == 4:
            return np.float32
        
        elif self.value == 8:
            return np.float64
        
        