# Plots

#%% Import

from sph_util import SPH_Util

import matplotlib.pyplot as plt


# Output Results

util = SPH_Util()

fig, ax = plt.subplots()
util.plot(data, plt, ax)