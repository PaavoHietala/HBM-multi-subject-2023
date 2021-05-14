#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:31:30 2021

@author: hietalp2
"""

import mne
import os
import sys

# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from Core import mne_common, mne_inverse, utils, visualize

### Parameters ----------------------------------------------------------------

# Root data directory of the project
project_dir = '/m/nbe/scratch/megci/MFinverse/Classic/'

# Subjects' MRI location

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# List of subject names

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(22,25)) if idx not in exclude]

# Source point spacing for source space calculation

src_spacing = 'ico4'

# Which BEM model to use for forward solution, <subject name> + <bem_suffix>.fif

bem_suffix = '-1-shell-bem-sol'

# Which inversion method to use for source activity estimate

stc_method = 'eLORETA'

# Which task is currently investigated

task = 'f'

# Which stimuli to analyze

stimuli = ['sector' + str(num) for num in range(1,25)]

# List of raw rest files for covariance matrix and extracting sensor info

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + subject.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for subject in subjects]

# List of MEG/MRI coregistration files for forward solution

coreg_files = ['/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/' + subject + 
               '/mri/T1-neuromag/sets/COR-ahenriks.fif' for subject in subjects]

# List of evoked response files for source activity estimate

evoked_files = [project_dir + 'Data/Evoked/' + subject + '_f-ave.fif' for
                subject in subjects]

# List of colors for each stimulus label

colors = ['mistyrose', 'plum', 'thistle', 'lightsteelblue', 'lightcyan', 'lightgreen',
          'lightyellow', 'papayawhip', 'lightcoral', 'violet', 'mediumorchid', 'royalblue',
          'aqua', 'mediumspringgreen', 'khaki', 'navajowhite', 'red', 'purple',
          'blueviolet', 'blue', 'turquoise', 'lime', 'yellow', 'orange']

# List of stimuli that should show response on both hemispheres:

bilaterals = ['sector3', 'sector7', 'sector11', 'sector15', 'sector19', 'sector23']

# Overwrite existing files

overwrite = True

### Pipeline steps to run -----------------------------------------------------


steps = {'prepare_directories' :        True,
         'compute_source_space' :       True,
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : True,
         'compute_covariance_matrix' :  False,
         'construct_inverse_operator' : True,
         'estimate_source_timecourse' : True,
         'morph_to_fsaverage' :         True,
         'average_stcs_source_space' :  False,
         'label_peaks' :                False,
         'expand_peak_labels' :         False,
         'label_all_vertices' :         False}


### Run the pipeline ----------------------------------------------------------

# Prepare all needed directories for the data
if steps['prepare_directories']:
    utils.prepare_directories(project_dir)

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    mne_common.compute_source_space('fsaverage', project_dir, src_spacing, overwrite,
                                    add_dist = True)

# Following steps are run on per-subject basis
for idx, subject in enumerate(subjects):
    
    # Compute source spaces for subjects and save them in ../Data/src/
    if steps['compute_source_space']:
        mne_common.compute_source_space(subject, project_dir, src_spacing, overwrite,
                                        morph = True, add_dist = True)
    
    # Setup forward model based on FreeSurfer BEM surfaces
    if steps['calculate_bem_solution']:
        mne_common.calculate_bem_solution(subject, project_dir, overwrite)
    
    # Calculate forward solutions for the subjects and save them in ../Data/fwd/
    if steps['calculate_forward_solution']:
        bem = os.path.join(subjects_dir, subject, 'bem', subject + bem_suffix + '.fif')
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
        mne_inverse.estimate_source_timecourse(subject, project_dir, raw, src_spacing,
                                               stc_method, fname_evokeds, task, stimuli,
                                               overwrite)
    
    # Morph subject data to fsaverage
    if steps['morph_to_fsaverage']:
        mne_common.morph_to_fsaverage(subject, project_dir, src_spacing,
                                      stc_method, task, stimuli, overwrite)

# Following steps are run on averaged data or produce averaged data

# Average data from all subjects for selected task and stimuli
if steps['average_stcs_source_space']:
    utils.average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                                    task, stimuli, overwrite)
    
# Select peaks from all averaged stimuli and plot on fsaverage
if steps['label_peaks']:
    visualize.label_peaks(subjects, project_dir, src_spacing, stc_method, task,
                          stimuli, colors, overwrite)

# Select peaks from all averaged stimuli, grow them 7mm and plot on lh + rh
if steps['expand_peak_labels']:
    visualize.expand_peak_labels(subjects, project_dir, src_spacing, stc_method,
                                 task, stimuli, colors, overwrite,
                                 bilaterals = bilaterals)

# Label each vertex based on normalized stimulus data and plot on lh + rh
if steps['label_all_vertices']:
    visualize.label_all_vertices(subjects, project_dir, src_spacing, stc_method,
                                 task, stimuli, colors, overwrite)