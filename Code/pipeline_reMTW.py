#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline utilizing the Janati et al. (2020) reweighted Minimum Wasserstein Estimate
(MWE0.5 or reMTW).

Created on Tue Feb  2 15:31:30 2021

@author: hietalp2
"""

import mne
import os
import sys
import numpy as np
from datetime import datetime
# from joblib import Parallel, delayed

# Dirty hack to get the relative import from same dir to work
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from Core import mne_common, solvers, utils, visualize, reMTW

from groupmne import compute_group_inverse, prepare_fwds
from mutar.utils import  groundmetric

print(datetime.now().strftime("%D.%M.%Y %H:%M:%S"),
      "Started pipeline_reMTW with parameters", sys.argv[1:])

### Parameters -----------------------------------------------------------------

# Root data directory of the project, str

project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'

# Subjects' MRI location, str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# List of subject names, subjects 1-24 available ex. those in exclude,
# list of str 

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]

# Get foci and geodesics results from individual subject instead of average,
# str or None

solo_subject = 'MEGCI_S1'

# Source point spacing for source space calculation, str

src_spacing = 'ico4'

# Which BEM model to use for forward solution,
# <subject name> + <bem_suffix>.fif, str

bem_suffix = '-1-shell-bem-sol'

# Which inversion method to use for source activity estimate,
# only remtw tested, str

stc_method = 'remtw'

# Which task is currently investigated, str

task = 'f'

# Which stimuli to analyze, sectors 1-24 available, list of str

stimuli = ['sector' + str(num) for num in range(1,25)]

# List of raw rest files for covariance matrix and extracting sensor info,
# list of str

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + subject.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for subject in subjects]

# List of MEG/MRI coregistration files for forward solution, list of str

coreg_files = ['/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/' + subject + 
               '/mri/T1-neuromag/sets/COR-ahenriks.fif' for subject in subjects]

# List of evoked response files for source activity estimate, list of str

evoked_files = ['/m/nbe/scratch/megci/MFinverse/reMTW/Data/Evoked/' + subject
                + '_f-ave.fif' for subject in subjects]

# List of matplotlib colors for each stimulus label, list of str

colors = ['mistyrose', 'plum', 'thistle', 'lightsteelblue', 'lightcyan', 'lightgreen',
          'lightyellow', 'papayawhip', 'lightcoral', 'violet', 'mediumorchid', 'royalblue',
          'aqua', 'mediumspringgreen', 'khaki', 'navajowhite', 'red', 'purple',
          'blueviolet', 'blue', 'turquoise', 'lime', 'yellow', 'orange']
colors_ecc = ['blue'] * 8 + ['yellow'] * 8 + ['red'] * 8
colors_polar = ['cyan', 'indigo', 'violet', 'magenta', 'red', 'orange', 'yellow', 'green'] * 3

# Overwrite existing files, bool

overwrite = True

# List of stimuli that should show response on both hemispheres, list of str

bilaterals = ['sector3', 'sector7', 'sector11', 'sector15', 'sector19', 'sector23']

# How many active source point are we aiming for, int

target = 2

# Suffix to append to filenames, used to distinguish averages of N subjects
# Expected format is len(subjects)< optional text>, str

suffix = str(len(subjects)) + 'subjects'

# File containing V1 peak timings for each subject, if None start and stop times
# will be used for all subjects, str or None

timing_fpath = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/V1_medians_evoked.csv'

# Averaged subject counts for which the geodesic distances are tabulated,
# list of int

counts = [1, 5, 10, 15, 20]

# Check CLI arguments, override other settings

alpha = None
beta = None
tenplot = False
start = 0.08
stop = 0.08
concomitant = False
for arg in sys.argv[1:]:
    if arg.startswith('-stim='):
        stimuli = arg[6:].split(',')
        print("Solving for stimuli", stimuli)
    elif arg.startswith('-stimnum='):
        stimuli = ['sector' + str(i) for i in arg[9:].split(',')]
        print("Solving for stimuli", stimuli)
    elif arg.startswith('-alpha='):
        alpha = float(arg[7:])
    elif arg.startswith('-beta='):
        beta = float(arg[6:])
    elif arg.startswith('-target='):
        target = float(arg[8:])
    elif arg.startswith('-tenplot'):
        tenplot = True
    elif arg.startswith('-time='):
        times = arg[6:].split(',')
        start = float(times[0])
        if len(times) > 1:
            stop = float(times[1])
        else:
            stop = start
        timing_fpath = None
    elif arg.startswith('-suffix='):
        suffix = arg[8:]
    elif arg.startswith('-concomitant='):
        concomitant = (True if arg[13:].lower() == "true" else False)
    elif arg.startswith('-dir='):
        project_dir = arg[5:]
    else:
        print('Unknown argument: ' + arg)

### Pipeline steps to run ------------------------------------------------------

steps = {'prepare_directories' :        False,
         'compute_source_space' :       False,
         'restrict_src_to_label' :      False, # Not working, MWE needs continuous surfs
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : False,
         'compute_covariance_matrix' :  False,
         'estimate_source_timecourse' : False,
         'morph_to_fsaverage' :         False,
         'average_stcs_source_space' :  False,
         'label_peaks' :                False, # Not really useful
         'expand_peak_labels' :         False, # For intermediate plots only
         'label_all_vertices' :         False, # Broken
         'plot_eccentricity_foci' :     False,
         'plot_polar_foci' :            False,
         'tabulate_geodesics' :         True}

### Run the pipeline -----------------------------------------------------------

# Prepare all needed directories for the data
if steps['prepare_directories']:
    utils.prepare_directories(project_dir)

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    mne_common.compute_source_space('fsaverage', project_dir, src_spacing, overwrite,
                                    add_dist = True)

# Attempt to restrict source points to a label, doesn't work currently
if steps['restrict_src_to_label']:
    labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s')
    utils.restrict_src_to_label('fsaverage', project_dir, src_spacing, overwrite, labels)

# Following steps are run on per-subject basis
for idx, subject in enumerate(subjects):   

    # Compute source spaces for subjects and save them in ../Data/src/
    if steps['compute_source_space']:
        mne_common.compute_source_space(subject, project_dir, src_spacing, overwrite,
                                        morph = True, add_dist = True)
    
    # Calculate 1-shell and 3-shell BEM solutions using FreeSurfer BEM surfaces
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

# Following steps are run on simulaneously for all subjects --------------------

# Estimate source timecourses
if steps['estimate_source_timecourse'] or tenplot == True:
    # Prepare list of evoked responses
    print("Loading evokeds")
    evokeds = []
    for fpath in evoked_files:
        evokeds.append(mne.read_evokeds(fpath, verbose = False))
    
    # Rearrange to stimulus-based listing instead of subject-based
    evokeds = [[evokeds_subj[i] for evokeds_subj in evokeds] for i in range(len(evokeds[0]))]
    
    # Load forward operators and noise covs
    print("Loading data for inverse solution")
    fwds_ = []
    noise_covs = []
    for idx, subject in enumerate(subjects):
        fname_fwd = utils.get_fname(subject, 'fwd', src_spacing = src_spacing)
        fpath_fwd = os.path.join(project_dir, 'Data', 'fwd', fname_fwd)
        fwds_.append(mne.read_forward_solution(fpath_fwd, verbose = False))
        
        fname_cov = utils.get_fname(subject, 'cov', fname_raw = rest_raws[idx])
        noise_cov = mne.read_cov(os.path.join(project_dir, 'Data', 'cov', fname_cov),
                                 verbose = False)
        noise_covs.append(noise_cov)

    # Load fsaverage source space for reference
    fname_ref = utils.get_fname('fsaverage', 'src', src_spacing = src_spacing)
    fpath_ref = os.path.join(project_dir, 'Data', 'src', fname_ref)
    
    src_ref = mne.read_source_spaces(fpath_ref)
        
    # Prepare forward operators for the inversion
    fwds = prepare_fwds(fwds_, src_ref, copy = False)

    # Load V1 peak timing for all subjects
    if timing_fpath != None:
        timing = np.loadtxt(timing_fpath).tolist()

        # Restrict timings only to selected subjects
        [timing.insert(i - 1, 0) for i in exclude]
        starts = [t for i, t in enumerate(timing) if str(i + 1)
                  in [sub[7:] for sub in subjects]]
        stops = starts.copy()
        print('Loaded timings: ', starts)
    else:
        starts = [start] * len(subjects)
        stops = [stop] * len(subjects)

    # Solve the inverse problem for each stimulus
    for stim in stimuli:
        print("Solving for stimulus " + stim)
        stim_idx = int("".join([i for i in stim if i in "1234567890"])) - 1
        evokeds = [ev.crop(starts[i], stops[i]) for i, ev in enumerate(evokeds[stim_idx])]

        # Change the Evoked timestamps from subject-specific to 0 to circumvent
        # a limitation in GroupMNE inverse.py, line 77 if subject-specific timing
        # is used
        if len(set(starts)) > 1:
            for ev in evokeds:
                ev.times = np.array([0.])

        print(starts, stops)
        print(evokeds)

        info = '-'.join([src_spacing, "subjects=" + str(len(subjects)), task, stim,
                        "target=" + str(target)])

        if stim in bilaterals:
            target *= 2

        if tenplot == True:
            reMTW.reMTW_tenplot_a(fwds, evokeds, noise_covs, stim, project_dir,
                                  concomitant = concomitant, beta = beta)
            reMTW.reMTW_tenplot_b(fwds, evokeds, noise_covs, stim, project_dir,
                                  concomitant = concomitant, alpha = alpha)
            continue

        solvers.group_inversion(subjects, project_dir, src_spacing, stc_method,
                                task, stim, fwds, evokeds, noise_covs, target,
                                overwrite, concomitant = concomitant, alpha = alpha,
                                beta = beta, info = info, suffix = suffix)

# Morph subject data to fsaverage
if steps['morph_to_fsaverage']:
    print('morphing to fsaverage')
    for subject in subjects:
        mne_common.morph_to_fsaverage(subject, project_dir, src_spacing,
                                      stc_method, task, stimuli, overwrite,
                                      suffix = suffix)

# Average data from all subjects for selected task and stimuli
if steps['average_stcs_source_space']:
    print('Averaging stcs in source space')
    utils.average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                                    task, stimuli, overwrite = overwrite,
                                    suffix = suffix)
    
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

# Plot all stimulus peaks on fsaverage LH and RH, color based on 3-ring eccentricity
if steps['plot_eccentricity_foci']:
    if solo_subject == None:
        visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                            colors_ecc, bilaterals, suffix, 'ecc', overwrite)
    else:
        visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                            colors_ecc, bilaterals, suffix, 'ecc', overwrite,
                            subject = solo_subject, stc_type = 'stc_m')

# Plot all stimulus peaks on fsaverage LH and RH, color based on wedge
if steps['plot_polar_foci']:
    if solo_subject == None:
        visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                            colors_polar, bilaterals, suffix, 'polar', overwrite)
    else:
        visualize.plot_foci(project_dir, src_spacing, stc_method, task, stimuli,
                            colors_polar, bilaterals, suffix, 'polar', overwrite,
                            subject = solo_subject, stc_type = 'stc_m')

# Tabulate geodesic distances between peaks and V1 on 1-20 averaged subjects
if steps['tabulate_geodesics']:
    if solo_subject == None:
        utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task,
                                stimuli, bilaterals, suffix, overwrite,
                                counts = counts)
    else:
        utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task,
                                stimuli, bilaterals, suffix, overwrite,
                                counts = counts, subject = 'MEGCI_S1',
                                mode = 'stc')
