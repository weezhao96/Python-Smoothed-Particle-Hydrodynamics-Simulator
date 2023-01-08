# Plots

#%% Import

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
ts = []

names = ['n','rho','p','x_0','x_1','v_0','v_1','a_0','a_1','empty']

for file in files:
    
    df = pd.read_csv(file, skiprows=1, sep=',')
    df.columns = names
    
    dfs.append(df)

for file in files:
    
    with open(file, 'r') as file:
        
        line = file.readline()
        ts.append(float(line[3:-2]))
        

fig, ax = plt.subplots()

i = 0

for df in dfs:

    plt.cla()

    ax.plot(df.x_0, df.x_1, '.', markersize=10) 
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    
    plt.xlabel('$x$', fontsize=15, usetex=False)
    plt.ylabel('$y$', fontsize=15, usetex=False)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title('t = {}'.format(ts[i]))

    plt.show()
    
    plt.pause(0.1)
    
    i += 1
    
    
