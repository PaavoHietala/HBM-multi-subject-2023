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
EVOKED_DIR = "/m/nbe/scratch/megci/MFinverse/Evoked_mne/"
RESULT_DIR = "/m/nbe/scratch/megci/MFinverse/"

mne.set_config("SUBJECTS_DIR", ROOT_DIR + 'FS_Subjects_MEGCI/')
subject = 'MEGCI_S2'

raw = ROOT_DIR + 'MEG/megci_rawdata_mc_ic/' + subject.lower() + '_mc/run4_raw_tsss_mc_transOHP_blinkICremoved.fif'
evoked = mne.read_evokeds(EVOKED_DIR + subject + '_f-ave.fif')

#%% Visualize MRI and MEG alignment

trans = ROOT_DIR + 'FS_Subjects_MEGCI/' + subject + '/mri/T1-neuromag/sets/COR-ahenriks.fif'
mne.viz.plot_alignment(evoked[0].info, trans, dig=True, meg=["helmet", "sensors"],
                       subjects_dir = mne.get_config("SUBJECTS_DIR"), subject = subject)

#%% Compute source space

src = mne.setup_source_space(subject, spacing = 'oct6', add_dist='patch')

fig = mne.viz.plot_alignment(subject=subject, surfaces='white', coord_frame='head', src=src)

#%% Calculate forward solution

bem_path = mne.get_config("SUBJECTS_DIR") + subject + '/bem/' + subject + '-3-shell-bem-sol.fif'

# Calculate the forward solution
fwd = mne.make_forward_solution(raw, trans = trans, src = src, bem = bem_path,
                                meg = True, eeg = False, n_jobs = 1, verbose = True)

# Constrain the dipoles to surface normals
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori = True, force_fixed = True, use_cps = True, copy = True)

# Write the forward solution to disk
fwd_name = subject + '-fwd.fif'
mne.write_forward_solution(RESULT_DIR + 'forward_solutions/' + fwd_name, fwd, overwrite=True)

#%% Load forward solution

fwd_name = subject + '-fwd.fif'
fwd = mne.read_forward_solution(RESULT_DIR + 'forward_solutions/' + fwd_name)

#%% Plotting sensitivity maps

grad_map = mne.sensitivity_map(fwd_fixed, ch_type = 'grad', mode = 'fixed')
mag_map = mne.sensitivity_map(fwd_fixed, ch_type = 'mag', mode = 'fixed')

brain_sens = grad_map.plot(subjects_dir = mne.get_config("SUBJECTS_DIR"), clim = dict(lims=[0, 50, 100]))
brain_sens = mag_map.plot(subjects_dir = mne.get_config("SUBJECTS_DIR"), clim = dict(lims=[0, 50, 100]))
