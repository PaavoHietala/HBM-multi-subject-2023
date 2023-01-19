#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for estimating retinotopic responses using eLORETA and source-space
averaging.

Created on Tue Feb  2 15:31:30 2021

@author: hietalp2
"""

import mne
import os
import numpy as np

from Core import mne_common, mne_inverse, utils

### Parameters -----------------------------------------------------------------

# Root data directory of the project. Use a different folder for each
# pipeline, e.g. .../MFinverse/Classic/ and .../MFinverse/reMTW/

project_dir = '/m/nbe/scratch/megci/MFinverse/Classic/'

# Subjects' MRI location (=FreeSurfer file location), str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# Subject names, IDs 1-24 available, 4 subjects are excluded for data issues,
# list of str 

exclude = [5, 8, 13, 15]
subjects = [f'MEGCI_S{id}' for id in list(range(1, 25)) if id not in exclude]

# Source point spacing for source space calculation, str

src_spacing = 'ico4'

# Which BEM model to use for forward solution. Naming convention for BEM
# files is <subject name> + <bem_suffix>.fif as created in the step
# calculate_bem_solution, str

bem_suffix = '-1-shell-bem-sol'

# Which inversion method to use for source activity estimate, str
# Options: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'

stc_method = 'eLORETA'

# Which task is currently investigated. Used as a suffix in file names,
# can also be 'None' if task suffix is not desired. str

task = 'f'

# Which stimuli to analyze, sectors 1-24 available, list of str

stimuli = [f'sector{num}' for num in range(1, 25)]

# Suffix to append to filenames, used to distinguish averages of N subjects
# Expected format is len(subjects)<optional text>, str

suffix = str(len(subjects)) + 'subjects'

# List of raw rest files for covariance matrix and extracting sensor info,
# list of str

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + s.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for s in subjects]

# List of MEG/MRI coregistration files for forward solution, list of str

coreg_files = ['/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/' + s + 
               '/mri/T1-neuromag/sets/COR-ahenriks.fif' for s in subjects]

# List of evoked response files for source activity estimate, list of str

evoked_files = [project_dir + 'Data/Evoked/' + subject + '_f-ave.fif' for
                subject in subjects]

# File containing V1 peak timings for each subject, if None start and stop times
# (by default 0.08) will be used for all subjects, str or None

timing_fpath = project_dir + 'Data/plot/V1_medians_evoked.csv'

# List of stimuli that should show response on both hemispheres, list of str

bilaterals = ['sector3', 'sector7', 'sector11',
              'sector15', 'sector19', 'sector23']

# Overwrite existing files, bool

overwrite = True

# Number of subjects to compute the averages for, list of str

s_nums = [1, 5, 10, 15, 20]

### Pipeline steps to run ------------------------------------------------------


steps = {'prepare_directories' :        False,
         'compute_source_space' :       False,
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : False,
         'compute_covariance_matrix' :  False,
         'construct_inverse_operator' : False,
         'estimate_source_timecourse' : False,
         'morph_to_fsaverage' :         False,
         'average_stcs_source_space' :  False,
         'tabulate_geodesics' :         False} 


### Run the pipeline, no changes required beyond this line ---------------------

if timing_fpath != None:
    # Load V1 peak timing for all subjects
    timing = np.loadtxt(timing_fpath).tolist()

    # Restrict timings only to selected subjects
    [timing.insert(i - 1, 0) for i in exclude]
    timing = [t for i, t in enumerate(timing) if str(i + 1)
              in [sub[7:] for sub in subjects]]
    print('Loaded timings: ', timing)

# Prepare all needed directories for the data
if steps['prepare_directories']:
    utils.prepare_directories(project_dir)

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    mne_common.compute_source_space('fsaverage', project_dir, src_spacing,
                                    overwrite, add_dist = True)

# Faollowing steps are run on per-subject basis
for idx, subject in enumerate(subjects):
    
    # Compute source spaces for subjects and save them in ../Data/src/
    if steps['compute_source_space']:
        mne_common.compute_source_space(subject, project_dir, src_spacing,
                                        overwrite, morph = True, add_dist = True)
    
    # Setup forward model based on FreeSurfer BEM surfaces
    if steps['calculate_bem_solution']:
        mne_common.calculate_bem_solution(subject, overwrite)
    
    # Calculate forward solutions for the subjects and save them in ../Data/fwd/
    if steps['calculate_forward_solution']:
        bem = os.path.join(subjects_dir, subject, 'bem',
                           subject + bem_suffix + '.fif')
        raw = rest_raws[idx]
        coreg = coreg_files[idx]
        mne_common.calculate_forward_solution(subject, project_dir, src_spacing,
                                              bem, raw, coreg, overwrite)
        
    # Calculate noise covariance matrix from rest data
    if steps['compute_covariance_matrix']:
        raw = rest_raws[idx]
        mne_common.compute_covariance_matrix(subject, project_dir, raw, overwrite)
    
    # Construct inverse operator
    if steps['construct_inverse_operator']:
        raw = rest_raws[idx]
        mne_inverse.construct_inverse_operator(subject, project_dir, raw,
                                               src_spacing, overwrite)
    
    # Estimate source timecourses
    if steps['estimate_source_timecourse']:
        raw = rest_raws[idx]
        fname_evokeds = evoked_files[idx]
        mne_inverse.estimate_source_timecourse(subject, project_dir, raw,
                                               src_spacing, stc_method,
                                               fname_evokeds, task, stimuli,
                                               SNR = 2, overwrite = overwrite)
    
    # Morph subject data to fsaverage
    if steps['morph_to_fsaverage']:
        mne_common.morph_to_fsaverage(subject, project_dir, src_spacing,
                                      stc_method, task, stimuli, overwrite)

# Following steps are run on averaged data or produce averaged data

# Average data from all subjects for selected task and stimuli
if steps['average_stcs_source_space']:
    for num in s_nums:
        subjects_ = subjects[:num]
        suffix_ = f'{num}{suffix.lstrip("0123456789")}' 
        utils.average_stcs_source_space(subjects_, project_dir, src_spacing,
                                        stc_method, task, stimuli, suffix_,
                                        timing = timing, overwrite = overwrite)

# Tabulate geodesic distances between peaks and V1 on 1-20 averaged subjects
if steps['tabulate_geodesics']:
    utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task,
                             stimuli, bilaterals, suffix, overwrite = overwrite)
