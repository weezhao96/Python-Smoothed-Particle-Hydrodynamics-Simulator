# Plots

#%% Import

from sph_util import SPH_Util

import matplotlib.pyplot as plt
import pandas as pd
import glob


#%% Read and Plot File

root_path = '../output/states'

files = []

for file in glob.glob(root_path + '/*.txt'):
    files.append(file)

files.sort()

dfs = []

names = ['n','rho','p','x_0','x_1','v_0','v_1','a_0','a_1']

for file in files:
    dfs.append(pd.read_csv(file, skiprows=2, sep=',', header=0, names=names))

fig, ax = plt.subplots()

for df in dfs:

    plt.cla()

    ax.plot(df.x_0, df.x_1, '.', markersize=10) 
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    
    plt.xlabel('$x$', fontsize=15, usetex=False)
    plt.ylabel('$y$', fontsize=15, usetex=False)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
        
    plt.show()
    
    plt.pause(0.1)
