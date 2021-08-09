'''
Tabulate V1 timing medians and stds for appendix B in LaTeX format.
'''

import numpy as np

# CSV file with peak timings for all subjects and stimuli

times = np.genfromtxt('/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/V1_timing_evoked.csv',
                      delimiter = ',')

# Output file

outf = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/V1_timing_stat_table.txt'

times[times >= 0.1] = np.nan
times[times <= 0.06] = np.nan

stds = np.round(np.nanstd(times, axis = 0) * 1000, 1)
medians = (np.nanmedian(times, axis = 0) * 1000).astype(int)

subjects = ['Subject ' + str(i) for i in range(1, 21)]

with open(outf, 'w') as f:
    f.write('\\begin{table}[htb]\n\\centering\n\\begin{tabular}{c c}\n')
    f.write('\\begin{tabular}{|c|c|c|}\n\\cline{2-3}\n\\multicolumn{1}{c|}{} ')
    f.write('& median & std \\\\\n\\hline\n')

    for s, med, std in zip(subjects[0:10], medians[0:10], stds[0:10]):
        f.write(s + ' & ' + str(med) + ' & ' + str(std) + ' \\\\\n')
    
    f.write('\\hline\n\\end{tabular}\n\\quad\n\\begin{tabular}{|c|c|c|}'
            + '\n\\cline{2-3}\n\\multicolumn{1}{c|}{}& median & std \\\\\n\\hline\n')
    
    for s, med, std in zip(subjects[10:], medians[10:], stds[10:]):
        f.write(s + ' & ' + str(med) + ' & ' + str(std) + ' \\\\\n')

    f.write('\hline\n\\end{tabular}\n\\end{tabular}\n\\caption{Caption}'
            + '\n\\label{tab:my_label}\n\\end{table}')
