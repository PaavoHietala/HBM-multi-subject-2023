'''
Get variance explained for each MNE inversion from logged terminal output and
save it as .csv
'''

import numpy as np

slurm_dir = '/m/nbe/scratch/megci/MFinverse/Classic/Data/slurm_out/'
output = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/MNE_R2s.csv'
job_ids = ['60485871_4294967294', '60485003_4294967294']

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]
stimuli = ['sector' + str(num) for num in range(1,25)]

files = [slurm_dir + idx + '.out' for idx in job_ids]
R2s = np.zeros((len(stimuli), len(subjects)))

for fname in files:
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Subject:'):
                subject_id = subjects.index(line.split(':')[1].strip())
                stimulus_id = stimuli.index(line.split(':')[3].strip())
            elif line.startswith('    Explained'):
                R2s[stimulus_id, subject_id] = float(line.strip().split(' ')[2][:-1])

np.savetxt(output, R2s, delimiter = ',', fmt = '%.2f')