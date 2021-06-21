#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A very basic analysis starting from premade evoked reponses & BEM models.

Inversion with MNE and no averaging/group analysis

Created on Fri Jan 29 14:55:32 2021

@author: hietalp2
"""

import mne

ROOT_DIR = "/m/nbe/scratch/megci/data/"
EVOKED_DIR = "/m/nbe/scratch/megci/MFinverse/Classic/Data/Evoked/"
RESULT_DIR = "/m/nbe/scratch/megci/MFinverse/"

mne.set_config("SUBJECTS_DIR", ROOT_DIR + 'FS_Subjects_MEGCI/')
subject = 'MEGCI_S1'

raw = ROOT_DIR + 'MEG/megci_rawdata_mc_ic/' + subject.lower() + '_mc/run4_raw_tsss_mc_transOHP_blinkICremoved.fif'
rest = ROOT_DIR + 'MEG/megci_rawdata_mc_ic/' + subject.lower() + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif'

evoked = mne.read_evokeds(EVOKED_DIR + subject + '_f-ave.fif') 

#%% Visualize MRI and MEG alignment

trans = ROOT_DIR + 'FS_Subjects_MEGCI/' + subject + '/mri/T1-neuromag/sets/COR-ahenriks.fif'
mne.viz.plot_alignment(evoked[0].info, trans, dig=False, meg=["helmet", "sensors"],
                       subjects_dir = mne.get_config("SUBJECTS_DIR"), subject = subject, surfaces = {'brain' : 0, 'head' : 1})

#%% Compute source space
spacing = 'oct6'
src = mne.setup_source_space(subject, spacing = spacing, add_dist='patch')

fig = mne.viz.plot_alignment(subject=subject, surfaces='white', coord_frame='head', src=src)

#%% Calculate forward solution

bem_path = mne.get_config("SUBJECTS_DIR") + subject + '/bem/' + subject + '-3-shell-bem-sol.fif'

# Calculate the forward solution
fwd = mne.make_forward_solution(raw, trans = trans, src = src, bem = bem_path,
                                meg = True, eeg = False, n_jobs = 1, verbose = True)

# Write the forward solution to disk
fwd_name = subject + '-' + spacing + '-fwd.fif'
mne.write_forward_solution(RESULT_DIR + 'forward_solutions/' + fwd_name, fwd, overwrite=True)

#%% Load forward solution

spacing = 'oct6'
fwd_name = subject + '-' + spacing + '-fwd.fif'
fwd = mne.read_forward_solution(RESULT_DIR + 'forward_solutions/' + fwd_name)

# Constrain the dipoles to surface normals
mne.convert_forward_solution(fwd, surf_ori = True, use_cps = True, copy = False)

#%% Plotting sensitivity maps

grad_map = mne.sensitivity_map(fwd, ch_type = 'grad', mode = 'fixed')
mag_map = mne.sensitivity_map(fwd, ch_type = 'mag', mode = 'fixed')

brain_sens = grad_map.plot(subjects_dir = mne.get_config("SUBJECTS_DIR"), clim = dict(lims=[0, 50, 100]))
brain_sens = mag_map.plot(subjects_dir = mne.get_config("SUBJECTS_DIR"), clim = dict(lims=[0, 50, 100]))

#%% Inverse modeling

# Compute covariance matrix for the subject
noise_cov = mne.compute_raw_covariance(mne.io.Raw(rest))

# Construct the inverse operator
inv = mne.minimum_norm.make_inverse_operator(evoked[19].info, fwd, noise_cov, loose = 0.2, depth = 0.8)

source_est = mne.minimum_norm.apply_inverse(evoked[19], inv, method = 'MNE')