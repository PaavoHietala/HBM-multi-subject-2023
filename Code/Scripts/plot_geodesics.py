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

# Base dir 

basedir = '/m/nbe/scratch/megci/MFinverse/'

# List of geodesic distance input files to plot

inputfs = [basedir + 'Classic/Data/plot/distances.csv',
           basedir + 'reMTW/Data/plot/distances_MEGCI_S1_stc_m.csv',
           basedir + 'reMTW/Data/plot/distances_fsaverage_avg.csv']

# List of output files (graphs)

outputfs = [basedir + 'Classic/Data/plot/geodesics_eLORETA.pdf',
           basedir + 'reMTW/Data/plot/geodesics_MEGCI_S1_stc_m.pdf',
           basedir + 'reMTW/Data/plot/geodesics_fsaverage_avg.pdf']

# List of latex table output files (mean/median/std statistics)

statf = basedir + 'reMTW/Data/plot/geodesics_stats_all.txt'

# Plot titles

titles = ['eLORETA & AVG', 'MWE single subject', 'MWE & AVG']

# Row titles, usually numbers of averaged subjects for each row 

rows = ['1', '5', '10', '15', '20']

# y-axis label

ylabels = ['Included subjects', 'Included subjects', 'Included subjects']

# Run the code -----------------------------------------------------------------
#

def geo_plot(inputf, title, ylabel, outputf, rows):
    '''
    Plots geodesic distances and medians to a nice dippa-ready plot.

    Parameters
    ----------
    inputf : str
        Path to input file to plot, expected CSV with 1 row per N_subjects
    title : str
        Title of the plot
    ylabel : str
        Ylabel of the plot
    outputf : str
        Path to output file, pdf preferred
    rows : list of str
        Row descriptions, usually number of subjects ["1", "5", ...]
    '''
    # Load data to a dataframe, reformat from wide to tall format
    dist = np.loadtxt(inputf, delimiter = ',')
    df = pd.DataFrame(dist.T, columns = rows)
    df = df.reset_index()
    df = pd.melt(df, value_vars = rows)

    # Replace inf with nearest even 10 to get them cleanly on the right side
    df['value'].where(df['value'] <= 100, np.inf, inplace = True)
    inf_value = 90 #round(df.loc[df['value'] != np.inf, 'value'].max() * 0.12) * 10
    df = df.replace(np.inf, inf_value)

    # Plot individual stimuli and means on the same image
    plt.interactive(True)
    fig = plt.figure(figsize = (3,4))

    # Set seed to get same jitter every time
    np.random.seed(1893)
    sns.stripplot(x = df['value'], y = df['variable'], jitter = 0.25,
                alpha = 0.5, size = 6)

    dist[dist == np.inf] = np.nan
    dist[dist >= 100] = np.nan
    medians = np.nanmedian(dist, axis = 1)
    print(medians)
    
    sns.stripplot(x = medians, y = rows, jitter = 0., alpha = 0.75, size = 12,
                marker = "P", linewidth = 1.5)

    # Set labels, ticks and other visuals
    plt.grid(True, axis = 'x')
    plt.xlabel("Geodesic distance (mm)", fontsize = font)
    plt.ylabel(ylabel, fontsize = font)
    start, end = fig.axes[0].get_xlim()
    ticks = np.arange(20, 100, 10)
    fig.axes[0].xaxis.set_ticks(ticks)

    plt.title(title)

    lbl = [str(tick) for tick in ticks]
    lbl[-1] = ">100"#r"$\infty$"
    fig.axes[0].set_xticklabels(lbl)
    plt.yticks(fontsize = font)
    plt.tight_layout()

    # Adjust the infinity sign size and vertical location
    # fig.axes[0].get_xticklabels()[-1].set_fontsize(20)
    # fig.axes[0].get_xaxis().get_major_ticks()[-1].set_pad(-1)

    plt.savefig(outputf)
    
def stats(inputfs, statf, rows, titles):
    '''
    Solve basic statistics for input files and output them to a .txt in latex
    tabular format.

    Parameters
    ----------
    inputfs : list of str
        List of input files to analyze, csv with one row per N_subjects expected
    statf : str
        Path to output file
    rows : list of str
        Row descriptions, usually number of subjects ["1", "5", ...]
    titles : list of str
        Titles for each inputfs, same as geodesic plot title
    '''

    stats = []
    infs = []

    # Load all inputfs to one 3D list
    for f in inputfs:
        dist = np.loadtxt(f, delimiter = ',')
        dist[dist == np.inf] = np.nan
        dist[dist >= 100] = np.nan

        medians = np.nanmedian(dist, axis = 1)
        means = np.nanmean(dist, axis = 1)
        stds = np.nanstd(dist, axis = 1)
        inf = np.count_nonzero(np.isnan(dist), axis = 1)

        stats.append([means, medians, stds])
        infs.append(inf)
    
    # Fit the stats list to numpy array, one row in numpy = one row in table
    # One row consist of mean, median, std * len(inputfs)
    stat_arr = np.zeros((len(rows), len(inputfs) * 3))

    for row_i in range(len(rows)):
        for file_i in range(len(inputfs)):
            for stat_i in range(3):
                stat_arr[row_i, file_i * 3 + stat_i] = round(stats[file_i][stat_i][row_i], 1)

    # Write a latex-ready table of the mean, median, std data and another table
    # to same file with inf
    with open(statf, 'w+') as f:
        f.write('\\begin{center}\n\\begin{tabular}{ |c|c|c|c|c|c|c|c|c|c| }\n')
        f.write('\\cline{2-10}\n \multicolumn{1}{c|}{} & '
                + ' & '.join(['\\multicolumn{3}{|c|}{'
                + t.replace('&', '\\&') + '}' for t in titles]) + '\\\\')
        f.write('\n\\hline\n $N$ & ' + ' & '.join(['mean', 'mdn', 'std'] * 3)
                + '\\\\' + '\n\\hline\n')
        for row_i, row in enumerate(rows):
            f.write('& '.join([row] + [str(i) for i in stat_arr[row_i].tolist()]) + '\\\\\n')
        f.write('\\hline\n\\end{tabular}\n\\end{center}\n\n')

        # inf table
        f.write('\\begin{center}\n\\begin{tabular}{' + '|c' * (len(rows) + 1) + '| }\n')
        f.write('\\hline\n $N$ & ' + ' & '.join(rows) + '\\\\\n\hline\n')
        for row_i, row in enumerate(infs):
            f.write(titles[row_i].replace('&', '\\&') + ' & '
                    + ' & '.join([str(r) for r in row]) + '\\\\\n')
        f.write('\\hline\n\\end{tabular}\n\\end{center}\n')   

# Plot geodesic distances
for inputf, title, ylabel, outputf, rows in zip(inputfs, titles, ylabels,
                                                outputfs, [rows] * 3):
    #continue
    geo_plot(inputf, title, ylabel, outputf, rows)

# Output stats in a latex-table
stats(inputfs, statf, rows, titles)
