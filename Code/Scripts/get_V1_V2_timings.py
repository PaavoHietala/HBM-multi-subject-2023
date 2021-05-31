'''
Extract V1 and V2 peaks in ms from MNE source estimates.

Output is a csv file with n_stimuli rows and n_subjects columns.
'''

import mne
import numpy as np
import sys
import os
# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Core import mne_common as mne_op

# Settings ---------------------------------------------------------------------

stc_dir = '/scratch/nbe/megci/MFinverse/Classic/Data/stc/'
evoked_dir = '/scratch/nbe/megci/MFinverse/Classic/Data/Evoked/'
output_dir = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/'
src_spacing = 'ico4'
stc_method = 'eLORETA'
task = 'f'
mode = 'evoked'

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]
stimuli = ['sector' + str(num) for num in range(1,25)]

# Execution --------------------------------------------------------------------

def stc_timing(stc_dir, src_spacing, stc_method, task, subjects, stimuli):
    '''
    Export peak timing from MNE.SourceEstimate structures saved in files.

    V1 is expected to be responsible for the peak between 60 and 100ms.
    V2 is expected to be responsible for the peak between 100 and 150ms.

    V1 is somewhat reliable, V2 is not really.

    Parameters
    ----------
    stc_dir : str
        Directory where the stc .fif files are saved in
    src_spacing : str
        Source space spacing used in the .stc files, e.g. 'ico4'
    stc_method : str
        Source estimation method used in the .stc files, e.g. 'eLORETA'
    task : str
        String describing the task, assumed to be in Evoked filename
    subjects : list of str
        List of subject names. One Evoked file per subject assumed
    stimuli : list of str
        Stimulus names / codes

    Returns
    -------
    Tuple of two numpy arrays (V1, V2)
        V1 and V2 timings in numpy arrays of form [n_stimuli, n_subjects]
    '''

    V1 = np.zeros((len(stimuli), len(subjects)))
    V2 = np.zeros((len(stimuli), len(subjects)))

    for sub_idx, subject in enumerate(subjects):
        print("Extracting timings for " + subject)

        # Prepare V1 and V2 labels for lh and rh
        v1_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                                + subject + '/label/lh.V1_exvivo.label', subject)
        v1_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                                + subject + '/label/rh.V1_exvivo.label', subject)
        v2_lh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                                + subject + '/label/lh.V2_exvivo.label', subject)
        v2_rh = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                                + subject + '/label/rh.V2_exvivo.label', subject)
        
        for stim_idx, stim in enumerate(stimuli):
            # Load stc and crop it to temporal range of expected max V1 and V2 contribution
            fname = stc_dir + mne_op.get_fname(subject, 'stc', stc_method = stc_method,
                                                src_spacing = src_spacing, task = task,
                                                stim = stim)
            stc = mne.read_source_estimate(fname)
            stc_V1 = stc.copy().crop(tmin = 0.06, tmax = 0.10)
            stc_V2 = stc.copy().crop(tmin = 0.10, tmax = 0.15)

            # Check V1 peak and save to array
            if np.max(stc_V1.in_label(v1_lh).lh_data) > np.max(stc_V1.in_label(v1_rh).rh_data):
                V1[stim_idx, sub_idx] = stc_V1.in_label(v1_lh).get_peak()[1]
            else:
                V1[stim_idx, sub_idx] = stc_V1.in_label(v1_rh).get_peak()[1]
            
            # Check V2 peak and save to array
            if np.max(stc_V2.in_label(v2_lh).lh_data) > np.max(stc_V2.in_label(v2_rh).rh_data):
                V2[stim_idx, sub_idx] = stc_V2.in_label(v2_lh).get_peak()[1]
            else:
                V2[stim_idx, sub_idx] = stc_V2.in_label(v2_rh).get_peak()[1]
            
            print(subject, stim, '%.3f' % V1[stim_idx, sub_idx],
                  '%.3f' % V2[stim_idx, sub_idx])
        
        return (V1, V2)

def evoked_timing(evoked_dir, task, subjects, stimuli):
    '''
    Export peak timing from MNE.Evoked structures saved in files.

    V1 is expected to be responsible for the peak between 60 and 100ms.
    V2 is expected to be responsible for the peak between 100 and 150ms.

    V1 is somewhat reliable, V2 is not really.

    Parameters
    ----------
    evoked_dir : str
        Directory where the Evoked .fif files are saved in
    task : str
        String describing the task, assumed to be in Evoked filename
    subjects : list of str
        List of subject names. One Evoked file per subject assumed
    stimuli : list of str
        Stimulus names / codes

    Returns
    -------
    Tuple of two numpy arrays (V1, V2)
        V1 and V2 timings in numpy arrays of form [n_stimuli, n_subjects]
    '''

    V1 = np.zeros((len(stimuli), len(subjects)))
    V2 = np.zeros((len(stimuli), len(subjects)))

    for sub_idx, subject in enumerate(subjects):
        print("Extracting timings for " + subject)

        fname = evoked_dir + subject + '_' + task + '-ave.fif'
        evokeds = mne.read_evokeds(fname, verbose = False)

        for evoked_idx, evoked in enumerate(evokeds):
            # Check V1 peak and save to array
            V1[evoked_idx, sub_idx] = evoked.get_peak(tmin = 0.06,
                                                      tmax = 0.10,
                                                      ch_type = 'mag')[1]
            
            # Check V2 peak and save to array
            V2[evoked_idx, sub_idx] = evoked.get_peak(tmin = 0.10,
                                                      tmax = 0.15,
                                                      ch_type = 'mag')[1]
            
            print(subject, evoked.comment, '%.3f' % V1[evoked_idx, sub_idx],
                  '%.3f' % V2[evoked_idx, sub_idx])
        
    return (V1, V2)

def medians(times, start = None, stop = None):
    '''
    Calculate median timing for each subject, exclude times <= start and >= stop

    Parameters
    ----------
    times : np.array
        Timings of all subjects and stimuli in [stimulus, subject] format
    start : float, optional
        Start of accepted value interval, by default None
    stop : float, optional
        Stop of accepted value interval, by default None
    
    Returns
    -------
    timing : np.array
        Median peak times for each subject
    '''

    times[times >= stop] = np.nan
    times[times <= start] = np.nan

    return np.nanmedian(times, axis = 0)

if mode == 'stc':
    V1, V2 = stc_timing(stc_dir, src_spacing, stc_method, task, subjects, stimuli)
elif mode == 'evoked':
    V1, V2 = evoked_timing(evoked_dir, task, subjects, stimuli)

# Output V1 and V2 arrays to csv files
np.savetxt(output_dir + 'V1_timing_' + mode + '.csv', V1, delimiter = ',', fmt = '%.3f')
np.savetxt(output_dir + 'V2_timing_' + mode + '.csv', V2, delimiter = ',', fmt = '%.3f')

# Calculate subject-specific medians and save them in separate file
V1_medians = medians(V1, start = 0.060, stop = 0.100)
np.savetxt(output_dir + 'V1_medians_' + mode + '.csv', V1_medians,
           delimiter = ',', fmt = '%.3f')