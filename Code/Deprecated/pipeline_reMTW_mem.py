#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:31:30 2021

Modification of the reMTW pipeline to keep everything in memory

@author: hietalp2
"""

import mne
import os
import sys
import numpy as np

# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from MNE_operations import mne_operations as mne_op

from groupmne import compute_group_inverse, prepare_fwds
from mutar.utils import  groundmetric

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

stc_method = 'remtw_mem'

# Which task is currently investigated

task = 'f'

# Which stimuli to analyze

stimuli = ['sector' + str(num) for num in range(9,17)]

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

# Overwrite existing files

overwrite = True

### Pipeline steps to run -----------------------------------------------------


steps = {'prepare_directories' :        False,
         'compute_source_space' :       True,
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : True,
         'compute_covariance_matrix' :  True,
         'estimate_source_timecourse' : True,
         'morph_to_fsaverage' :         False,
         'average_stcs_source_space' :  False,
         'label_peaks' :                False,
         'expand_peak_labels' :         False,
         'label_all_vertices' :         False}


### Run the pipeline ----------------------------------------------------------

# Prepare all needed directories for the data
if steps['prepare_directories']:
    for dirname in ['Data',
                    'Data/fwd',
                    'Data/src',
                    'Data/inv',
                    'Data/stc',
                    'Data/stc_m',
                    'Data/avg',
                    'Data/labels',
                    'Data/cov',
                    'Data/plot']:
        try:
            os.makedirs(project_dir + dirname)
        except FileExistsError:
            pass

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    src_ref = mne.setup_source_space('fsaverage', spacing = src_spacing, add_dist=False)

# Following steps are run on per-subject basis
srcs = []
bems = []
fwds = []
covs = []
for idx, subject in enumerate(subjects):
    print(subject)
    
    # Compute source spaces for subject
    if steps['compute_source_space']:
        src = mne.morph_source_spaces(src_ref, subject_to=subject)
        srcs.append(src)
    
    # Setup forward model based on FreeSurfer BEM surfaces
    if steps['calculate_bem_solution']:
        #bem = mne.make_bem_model(subject, ico = None, conductivity = [0.3])
        #bem = mne.make_bem_solution(bem)
        #bems.append(bem)
        mne_op.calculate_bem_solution(subject, project_dir, overwrite)
    
    # Calculate forward solutions for the subjects and save them in ../Data/fwd/
    if steps['calculate_forward_solution']:
        raw = rest_raws[idx]
        coreg = coreg_files[idx]
        bem = os.path.join(subjects_dir, subject, 'bem', subject + bem_suffix + '.fif')
        print(raw)
        print(coreg)
        print(src)
        print(bem)
        fwd = mne.make_forward_solution(raw, trans = coreg, src = src, bem = bem,
                                        meg = True, eeg = False)
        fwds.append(fwd)
        
    # Calculate noise covariance matrix from rest data
    if steps['compute_covariance_matrix']:
        raw = rest_raws[idx]
        noise_cov = mne.compute_raw_covariance(mne.io.Raw(raw))
        covs.append(noise_cov)

# Following steps are run on simulaneously for all subjects

# Estimate source timecourses
if steps['estimate_source_timecourse']:
    # Prepare list of evoked responses
    print("Loading evokeds")
    evokeds = []
    for fpath in evoked_files:
        evokeds.append(mne.read_evokeds(fpath, verbose = False))
    
    # Rearrange to stimulus-based listing instead of subject-based
    evokeds = [[evokeds_subj[i] for evokeds_subj in evokeds] for i in range(len(evokeds[0]))]

    #print('Defining base M')
    #base_M = groundmetric(fwds[0]['nsource'], p=2, normed = True)
        
    # Prepare forward operators for the inversion
    fwds_ = prepare_fwds(fwds, src_ref, copy = False)
        
    # Solve the inverse problem for each stimulus with reMTW
    for stim_idx, stim in enumerate(stimuli):
        print("Solving for stimulus " + stim)
        '''
        # Check that stcs for all stimuli have been calculated and saved
        comp = False
        for subject in subjects:
            fname_stc = mne_op.get_fname(subject, 'stc', src_spacing = src_spacing,
                                         stc_method = stc_method, task = task, stim=stim)
            fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
            if not os.path.isfile(fpath_stc + '-lh.stc'):
                print(fpath_stc + " Doesn't exist")
                comp = True
                break
        
        if overwrite or comp:  '''     
        # Calculate the inverse solutions for all subjects simultaneously
        evokeds_stimx = [ev.crop(0.08,0.08) for ev in evokeds[stim_idx]] # ev.crop(0.7, 0.9)
        stcs_remtw = compute_group_inverse(fwds_, evokeds_stimx, covs, #M=np.copy(base_M),
                                           method = 'remtw', alpha = 1, beta = 0.5, concomitant = True,
                                           n_jobs = 32, gpu = True, ot_threshold = 1e-7,
                                           max_iter_ot = 20, max_iter_cd = 10000, warm_start = True)
        
        # Save the returned stcs to disk
        #print(stcs_remtw)
        for i, stc in enumerate(stcs_remtw):
            fname_stc = mne_op.get_fname(subjects[i], 'stc', src_spacing = src_spacing,
                                         stc_method = stc_method, task = task, stim=stim)
            fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
            stc.save(fpath_stc)

# Morph subject data to fsaverage
if steps['morph_to_fsaverage']:
    print('morphing to fsaverage')
    for subject in subjects:
        mne_op.morph_to_fsaverage(subject, project_dir, src_spacing,
                                  stc_method, task, stimuli, overwrite)

# Average data from all subjects for selected task and stimuli
if steps['average_stcs_source_space']:
    print('Averaging stcs in source space')
    mne_op.average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                                     task, stimuli, overwrite)
    
# Select peaks from all averaged stimuli and plot on fsaverage
if steps['label_peaks']:
    mne_op.label_peaks(subjects, project_dir, src_spacing, stc_method, task,
                       stimuli, colors, overwrite)

# Select peaks from all averaged stimuli, grow them 7mm and plot on lh + rh
if steps['expand_peak_labels']:
    mne_op.expand_peak_labels(subjects, project_dir, src_spacing, stc_method,
                              task, stimuli, colors, overwrite)

# Label each vertex based on normalized stimulus data and plot on lh + rh
if steps['label_all_vertices']:
    mne_op.label_all_vertices(subjects, project_dir, src_spacing, stc_method,
                              task, stimuli, colors, overwrite)
