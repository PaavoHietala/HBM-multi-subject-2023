'''
Plot geodesic distances between peaks and a label from a file

Created on Thu May 27 15:22:35 2021

@author: hietalp2
'''

import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings ---------------------------------------------------------------------

# Font size for x and y labels

font = 11

# Directory inside which all actions are taken

project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/Data/plot/'

# Identifier string in filename between distances and csv

identifier = '_MEGCI_S1_stc' #'_fsaverage_avg'

# CSV File with all the geodesic distances in [nave, peak] format

input_f = 'distances' + identifier + '.csv'

# Row titles, usually numbers of averaged subjects for each row 

rows = ['1', '5', '10', '15', '20']

# y-axis label

ylabel = "Included subjects"

# Run the code -----------------------------------------------------------------
#
# Load data to a dataframe, reformat from wide to tall format
dist = np.loadtxt(project_dir + input_f, delimiter = ',')
df = pd.DataFrame(dist.T, columns = rows)
df = df.reset_index()
df = pd.melt(df, value_vars = rows)

# Replace inf with nearest even 10 to get them cleanly on the right side
#df['value'].where(df['value'] <= 150, np.inf, inplace = True)
inf_value = round(df.loc[df['value'] != np.inf, 'value'].max() * 0.12) * 10
df = df.replace(np.inf, inf_value)

# Plot individual stimuli and means on the same image
plt.interactive(True)
fig = plt.figure(figsize = (3,4))

sns.stripplot(x = df['value'], y = df['variable'], jitter = 0.25,
              alpha = 0.5, size = 6)

means = np.mean(np.ma.masked_invalid(dist), axis = 1).data
sns.stripplot(x = means, y = rows, jitter = 0., alpha = 0.75, size = 12,
              marker = "P", linewidth = 1.5)

# Set labels, ticks and other visuals
plt.grid(True, axis = 'x')
plt.xlabel("Geodesic distance (mm)", fontsize = font)
plt.ylabel(ylabel, fontsize = font)
start, end = fig.axes[0].get_xlim()
ticks = np.arange(10, math.ceil(end/10)*10, math.ceil((end - start) / 70) * 10)
fig.axes[0].xaxis.set_ticks(ticks)

lbl = [str(tick) for tick in ticks]
lbl[-1] = r"$\infty$"
fig.axes[0].set_xticklabels(lbl)
plt.yticks(fontsize = font)
plt.tight_layout()

# Adjust the infinity sign size and vertical location
fig.axes[0].get_xticklabels()[-1].set_fontsize(20)
fig.axes[0].get_xaxis().get_major_ticks()[-1].set_pad(-1)

plt.savefig(project_dir + 'geodesics' + identifier + '.pdf')
