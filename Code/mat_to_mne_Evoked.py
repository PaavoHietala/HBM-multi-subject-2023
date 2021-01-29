#%% Function to transform preprocessed .mat to mne.Evoked structure

import mne
import numpy as np
import scipy.io

def mat_to_mne_Evoked(mat_files, raw_files, results, arr_key = 'estimate',
                      tmin = 0, nave = 1):
    '''
    Converts MATLAB preprocessed .mat files of evoked responses to MNEpython's
    mne.Evoked class with sensor information from the original recording .fif.
    
    MATLAB files have to be of form stimulus x time x sensors.

    Parameters
    ----------
    mat_files : list
        List of full paths to convert
    raw_files:
        List of full paths to raw files from which to extract sensor info
    results : list
        List of full paths to output files
    arr_key : str
        Name of the array inside the .mat file, default: estimate
    tmin : float
        Start of the averaged response in relation to the stimulus, default: 0
    nave : int
        Number of averaged responses (epochs), default : 1

    Returns
    -------
    None.
    '''

    for i in range(len(mat_files)):
        # Load .mat data to an array and raw recording to mne.Raw
        mat = scipy.io.loadmat(mat_files[i])
        raw = mne.io.Raw(raw_files[i])
        output = results[i]

        # Remove all EOG and HPI channels as they are missing from matlab data
        raw.info.pick_channels([sensor['ch_name'] for sensor in raw.info['chs'] if
                                sensor['ch_name'][:3] == 'MEG'])
        
        # Collect evoked responses to each stimulus into a list
        evoked_list = []
        for stim_id, response in enumerate(mat[arr_key]):       
            evoked_array = mne.EvokedArray(np.transpose(response), raw.info,
                                           tmin = tmin, nave = nave,
                                           comment = 'Sector ' + str(stim_id+1))
             
            evoked_list.append(evoked_array)
        
        # Write one evoked .fif file per subject
        mne.write_evokeds(output, evoked_list)

#%% Example function call to transform a folder's worth of matlab-analyzed data

ROOT_DIR = "/m/nbe/scratch/megci/data/MEG/"
result_dir = "/m/nbe/scratch/megci/MFinverse/Evoked_mne/"

raw_files = []
mat_files = []
results = []

# Compile list of files to convert, raw data files and output files
for s_id in [s_id for s_id in list(range(1,25)) if s_id not in [5,8,13,15]]:
    raw_files.append(ROOT_DIR + 'megci_rawdata_mc_ic/megci_s' + str(s_id) +
                     '_mc/run4_raw_tsss_mc_transOHP_blinkICremoved.fif')
    mat_files.append(ROOT_DIR + 'evoked/S' + str(s_id) + '_f.mat')
    results.append(result_dir + 'MEGCI_S' + str(s_id) + '_f-ave.fif')

mat_to_mne_Evoked(mat_files, raw_files, results, tmin = -0.05, nave = 1806)