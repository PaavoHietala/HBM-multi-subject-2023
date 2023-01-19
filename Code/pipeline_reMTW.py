#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline utilizing the Janati et al. (2020) reweighted Minimum Wasserstein
Estimates (MWE0.5 or reMTW).

Created on Tue Feb  2 15:31:30 2021

@author: hietalp2
"""

import mne
import os
import sys
import numpy as np
from datetime import datetime

from Core import mne_common, solvers, utils, reMTW
from groupmne import prepare_fwds

print(datetime.now().strftime("%D.%M.%Y %H:%M:%S"),
      "Started pipeline_reMTW with parameters", sys.argv[1:])

### Parameters -----------------------------------------------------------------

# Root data directory of the project. Use a different folder for each
# pipeline, e.g. .../MFinverse/Classic/ and .../MFinverse/reMTW/

project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'

# Subjects' MRI location (=FreeSurfer file location), str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# Subject names, IDs 1-24 available, 4 subjects are excluded for data issues,
# list of str 

exclude = [5, 8, 13, 15]
subjects = [f'MEGCI_S{id}' for id in list(range(1, 25)) if id not in exclude]

# Get foci and geodesics results from individual subject ("Single-subject MWE"),
# e.g. 'MEGCI_S1'. None produces averaged results ("MWE with source-space
# averaging").

solo_subject = None

# Source point spacing for source space calculation, str

src_spacing = 'ico4'

# Which BEM model to use for forward solution. Naming convention for BEM
# files is <subject name> + <bem_suffix>.fif as created in the step
# calculate_bem_solution, str

bem_suffix = '-1-shell-bem-sol'

# Which groupmne inversion method to use for source activity estimate,
# only remtw tested, str

stc_method = 'remtw'

# Which task is currently investigated. Used as a suffix in file names,
# can also be 'None' if task suffix is not desired. str

task = 'f'

# Which stimuli to analyze, sectors 1-24 available, list of str

stimuli = [f'sector{num}' for num in range(1, 25)]

# List of raw rest files for covariance matrix and extracting sensor info,
# list of str

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + s.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for s in subjects]

# List of MEG/MRI coregistration files for forward solution, list of str

coreg_files = ['/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/' + s + 
               '/mri/T1-neuromag/sets/COR-ahenriks.fif' for s in subjects]

# List of evoked response files for source activity estimate, list of str

evoked_files = [project_dir + '/Data/Evoked/' + subject + '_f-ave.fif' for
                subject in subjects]

# Overwrite existing files, bool

overwrite = True

# List of stimuli that should show response on both hemispheres, list of str

bilaterals = ['sector3', 'sector7', 'sector11',
              'sector15', 'sector19', 'sector23']

# How many average active source points are we aiming for, int

target = 2

# Suffix to append to filenames, used to distinguish averages of N subjects
# Expected format is len(subjects)<optional text>, str

suffix = str(len(subjects)) + 'subjects'

# File containing V1 peak timings for each subject, if None start and stop times
# (by default 0.08) will be used for all subjects, str or None

timing_fpath = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/V1_medians_evoked.csv'

# Averaged subject counts for which the geodesic distances are tabulated,
# list of int

counts = [1, 5, 10, 15, 20]

### Pipeline steps to run ------------------------------------------------------

steps = {'prepare_directories' :        False,
         'compute_source_space' :       False,
         'calculate_bem_solution' :     False,
         'calculate_forward_solution' : False,
         'compute_covariance_matrix' :  False,
         'estimate_source_timecourse' : False,
         'morph_to_fsaverage' :         False,
         'average_stcs_source_space' :  False,
         'tabulate_geodesics' :         False}

### Run the pipeline -----------------------------------------------------------

# Check CLI arguments, override other settings

alpha = None
beta = None
hyper_plot = False
start = 0.08
stop = 0.08
concomitant = False

for arg in sys.argv[1:]:
    if arg.startswith('-stim='):
        stimuli = arg[6:].split(',')
        print("Solving for stimuli", stimuli)
    elif arg.startswith('-alpha='):
        alpha = float(arg[7:])
    elif arg.startswith('-beta='):
        beta = float(arg[6:])
    elif arg.startswith('-target='):
        target = float(arg[8:])
    elif arg.startswith('-hyperplot'):
        hyper_plot = True
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

# Prepare all needed directories for the data
if steps['prepare_directories']:
    utils.prepare_directories(project_dir)

# Compute source space for fsaverage before subjects
if steps['compute_source_space']:
    mne_common.compute_source_space('fsaverage', project_dir, src_spacing,
                                    overwrite, add_dist = True)

# Following steps are run on per-subject basis
for idx, subject in enumerate(subjects):   

    # Compute source spaces for subjects and save them in ../Data/src/
    if steps['compute_source_space']:
        mne_common.compute_source_space(subject, project_dir, src_spacing,
                                        overwrite, morph = True, add_dist = True)
    
    # Calculate 1-shell and 3-shell BEM solutions using FreeSurfer BEM surfaces
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

# Following steps are run on simulaneously for all subjects --------------------

# Estimate source timecourses
if steps['estimate_source_timecourse'] or hyper_plot == True:
    # Prepare list of evoked responses
    print("Loading evokeds")
    evokeds = []
    for fpath in evoked_files:
        evokeds.append(mne.read_evokeds(fpath, verbose = False))
    
    # Rearrange to stimulus-based listing instead of subject-based
    evokeds = [[e[i] for e in evokeds] for i in range(len(evokeds[0]))]
    
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
        evokeds = [ev.crop(starts[i], stops[i]) for i, ev
                   in enumerate(evokeds[stim_idx])]

        # Change the Evoked timestamps from subject-specific to 0 to circumvent
        # a limitation in GroupMNE inverse.py, line 77 if subject-specific
        # timing is used
        if len(set(starts)) > 1:
            for ev in evokeds:
                ev.times = np.array([0.])

        print(starts, stops)
        print(evokeds)

        info = '-'.join([src_spacing, "subjects=" + str(len(subjects)), task,
                         stim, "target=" + str(target)])

        if stim in bilaterals:
            target *= 2

        if hyper_plot == True:
            reMTW.reMTW_hyper_plot(fwds, evokeds, noise_covs, stim, project_dir,
                                   concomitant = concomitant, param = 'alpha',
                                   secondary = 0.3)
            reMTW.reMTW_hyper_plot(fwds, evokeds, noise_covs, stim, project_dir,
                                   concomitant = concomitant, param = 'beta',
                                   secondary = 7.5)
            continue

        solvers.group_inversion(subjects, project_dir, src_spacing, stc_method,
                                task, stim, fwds, evokeds, noise_covs, target,
                                overwrite, concomitant = concomitant,
                                alpha = alpha, beta = beta, info = info,
                                suffix = suffix)

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
    utils.average_stcs_source_space(subjects, project_dir, src_spacing,
                                    stc_method, task, stimuli,
                                    overwrite = overwrite, suffix = suffix)

# Tabulate geodesic distances between peaks and V1 on 1-20 averaged subjects
if steps['tabulate_geodesics']:
    if solo_subject == None:
        utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task,
                                 stimuli, bilaterals, suffix,
                                 overwrite = overwrite, counts = counts)
    else:
        utils.tabulate_geodesics(project_dir, src_spacing, stc_method, task,
                                 stimuli, bilaterals, suffix,
                                 overwrite = overwrite, counts = counts,
                                 subject = solo_subject, mode = 'stc_m')
