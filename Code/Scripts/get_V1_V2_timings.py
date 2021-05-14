'''
Extract V1 and V2 peaks in ms from MNE source estimates.
'''
import mne
import numpy as np
import sys
import os
# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Core import mne_common as mne_op

stc_dir = '/scratch/nbe/megci/MFinverse/Classic/Data/stc/'
output_dir = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/'
src_spacing = 'ico4'
stc_method = 'eLORETA'
task = 'f'

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]
stimuli = ['sector' + str(num) for num in range(1,25)]

V1 = np.zeros((len(stimuli), len(subjects)))
V2 = np.zeros((len(stimuli), len(subjects)))


for sub_idx, subject in enumerate(subjects):
    v1_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + subject + '/label/lh.V1_exvivo.label', subject)
    v1_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + subject + '/label/rh.V1_exvivo.label', subject)
    v2_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + subject + '/label/lh.V2_exvivo.label', subject)
    v2_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + subject + '/label/rh.V2_exvivo.label', subject)
    
    for stim_idx, stim in enumerate(stimuli):
        fname = stc_dir + mne_op.get_fname(subject, 'stc', stc_method = stc_method,
                                           src_spacing = src_spacing, task = task,
                                           stim = stim)
        stc = mne.read_source_estimate(fname)

        # Check V1 peak
        if np.max(stc.in_label(v1_lh).lh_data) > np.max(stc.in_label(v1_rh).rh_data):
            V1[stim_idx, sub_idx] = stc.in_label(v1_lh).get_peak()[1]
        else:
            V1[stim_idx, sub_idx] = stc.in_label(v1_rh).get_peak()[1]
        
        # Check V2 peak
        if np.max(stc.in_label(v2_lh).lh_data) > np.max(stc.in_label(v2_rh).rh_data):
            V2[stim_idx, sub_idx] = stc.in_label(v2_lh).get_peak()[1]
        else:
            V2[stim_idx, sub_idx] = stc.in_label(v2_rh).get_peak()[1]
        
        print(subject, stim, '%.3f' % V1[stim_idx, sub_idx],
              '%.3f' % V2[stim_idx, sub_idx])

np.savetxt(output_dir + 'V1_timing.csv', V1, delimiter = ',', fmt = '%.3f')
np.savetxt(output_dir + 'V2_timing.csv', V2, delimiter = ',', fmt = '%.3f')