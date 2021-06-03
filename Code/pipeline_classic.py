#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:31:30 2021

@author: hietalp2
"""

import mne
import os
import sys
import numpy as np

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
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]

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

# Suffix to append to filenames, used to distinguish averages of N subjects
# Expected format is len(subjects)< optional text>

suffix = str(len(subjects)) + 'subjects-avg-test'

# List of raw rest files for covariance matrix and extracting sensor info

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + subject.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for subject in subjects]

# List of MEG/MRI coregistration files for forward solution

coreg_files = ['/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/' + subject + 
               '/mri/T1-neuromag/sets/COR-ahenriks.fif' for subject in subjects]

# List of evoked response files for source activity estimate

evoked_files = [project_dir + 'Data/Evoked/' + subject + '_f-ave.fif' for
                subject in subjects]

# File containing V1 peak timings for each subject
timing_fpath = project_dir + 'Data/plot/V1_medians_evoked.csv'

# List of colors for each stimulus label, eccentricity rings and polar angles

colors = ['mistyrose', 'plum', 'thistle', 'lightsteelblue', 'lightcyan', 'lightgreen',
          'lightyellow', 'papayawhip', 'lightcoral', 'violet', 'mediumorchid', 'royalblue',
          'aqua', 'mediumspringgreen', 'khaki', 'navajowhite', 'red', 'purple',
          'blueviolet', 'blue', 'turquoise', 'lime', 'yellow', 'orange']
colors_ecc = ['blue'] * 8 + ['yellow'] * 8 + ['red'] * 8
colors_polar = ['cyan', 'indigo', 'violet', 'magenta', 'red', 'orange', 'yellow', 'green'] * 3

# List of stimuli that should show response on both hemispheres:

bilaterals = ['sector3', 'sector7', 'sector11', 'sector15', 'sector19', 'sector23']

# Overwrite existing files

overwrite = True

### Pipeline steps to run -----------------------------------------------------


steps = {'prepare_directories' :        False,
         'compute_source_space' :       False,
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : False,
         'compute_covariance_matrix' :  False,
         'construct_inverse_operator' : False,
         'estimate_source_timecourse' : False,
         'morph_to_fsaverage' :         False,
         'average_stcs_source_space' :  True,
         'label_peaks' :                False, # Not really useful
         'expand_peak_labels' :         False, # For intermediate plots only
         'label_all_vertices' :         False, # Broken
         'plot_eccentricity_foci' :     True,
         'plot_polar_foci' :            True,
         'tabulate_geodesics' :         False} 


### Run the pipeline ----------------------------------------------------------

# Prepare all needed directories for the data
if steps['prepare_directories']:
    utils.prepare_directories(project_dir)

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    mne_common.compute_source_space('fsaverage', project_dir, src_spacing, overwrite,
                                    add_dist = True)

# Faollowing steps are run on per-subject basis
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
    # Load V1 peak timing for all subjects
    timing = np.loadtxt(timing_fpath).tolist()

    # Restrict timings only to selected subjects
    [timing.insert(i - 1, 0) for i in exclude]
    timing = [t for i, t in enumerate(timing) if str(i + 1)
              in [sub[7:] for sub in subjects]]
    print('Loaded timings: ', timing)

    utils.average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                                    task, stimuli, suffix, timing = timing,
                                    overwrite = overwrite)
    
# Select peaks from all averaged stimuli and plot on fsaverage
if steps['label_peaks']:
    visualize.label_peaks(subjects, project_dir, src_spacing, stc_method, task,
                          stimuli, colors, overwrite)

# Select peaks from all averaged stimuli, grow them 7mm and plot on lh + rh
if steps['expand_peak_labels']:
    visualize.expand_peak_labels(subjects, project_dir, src_spacing, stc_method,
                                 task, stimuli, colors, suffix, overwrite,
                                 bilaterals = bilaterals)

# Label each vertex based on normalized stimulus data and plot on lh + rh
if steps['label_all_vertices']:
    visualize.label_all_vertices(subjects, project_dir, src_spacing, stc_method,
                                 task, stimuli, colors, overwrite)

# Plot all stimulus peaks on fsaverage LH and RH, color based on 3-ring eccentricity
if steps['plot_eccentricity_foci']:
    visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                        colors_ecc, bilaterals, suffix, 'ecc', overwrite)

# Plot all stimulus peaks on fsaverage LH and RH, color based on wedge
if steps['plot_polar_foci']:
    visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                        colors_polar, bilaterals, suffix, 'polar', overwrite)

# Tabulate geodesic distances between peaks and V1 on 1-20 averaged subjects
if steps['tabulate_geodesics']:
    utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task, stimuli,
                             bilaterals, suffix, overwrite)