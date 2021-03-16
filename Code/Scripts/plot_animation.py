import mne
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from MNE_operations import mne_operations as mne_op

### Parameters ----------------------------------------------------------------

# Root data directory of the project
project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'

# Subjects' MRI location

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# List of subject names

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]

# Source point spacing for source space calculation

src_spacing = 'ico4'

# Which BEM model to use for forward solution, <subject name> + <bem_suffix>.fif

bem_suffix = '-1-shell-bem-sol'

# Which inversion method to use for source activity estimate

stc_method = 'remtw'

# Which task is currently investigated

task = 'f'

# Which stimuli to analyze

stimuli = ['sector' + str(num) for num in range(14,15)]

# List of colors for each stimulus label

colors = ['mistyrose', 'plum', 'thistle', 'lightsteelblue', 'lightcyan', 'lightgreen',
          'lightyellow', 'papayawhip', 'lightcoral', 'violet', 'mediumorchid', 'royalblue',
          'aqua', 'mediumspringgreen', 'khaki', 'navajowhite', 'red', 'purple',
          'blueviolet', 'blue', 'turquoise', 'lime', 'yellow', 'orange']

# Overwrite existing files

overwrite = True

### ----------------------------------------------------------------------------
for stim in stimuli:
    for subject in subjects:
        fname_stc = mne_op.get_fname(subject, 'stc', src_spacing = src_spacing,
                                     stc_method = stc_method, task = task, stim = stim)
        fpath_plot = project_dir + '/Data/plot/' + fname_stc + '.gif'
        
        if not os.path.isfile(fpath_plot) or overwrite:
            print('Plotting ' + fpath_plot)
            stc = mne.read_source_estimate(project_dir + 'Data/stc/' + fname_stc)
            brain_kwargs = {'show' : False}
            #b = stc.plot('MEGCI_S12', 'inflated','lh', brain_kwargs = kwargs)
            #b.show_view({'elevation' : 100, 'azimuth' : -55}, distance = 500)
            b = mne.viz.plot_source_estimates(stc, subject, 'inflated', 'lh')
            b.save_movie(fpath_plot, time_dilation = 4)