#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Wrappers for different inverse problem solver functions.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import os
import copy
from .reMTW import reMTW_find_param, reMTW_wrapper
from .utils import get_fname

def group_inversion(subjects, project_dir, src_spacing, stc_method, task, stim,
                    fwds, evokeds, noise_covs, target, overwrite, info = '',
                    suffix = None, **solver_kwargs):
    '''
    Wrapper for groupMNE multi-subject solvers. Handles parameter search
    and saving the final source estimates to disk.

    Parameters
    ----------
    subjects : list of str
        List of subject identifiers/names.
    project_dir : str
        Base directory of the project.
    src_spacing : str
        Source spacing used for the source models, e.g. 'ico4'.
    stc_method : str
        Inverse solver used, e.g. 'reMTW'.
    task : str
        Task analyzed, e.g. 'f'.
    stim : str
        Stimulus name, e.g. 'sector22'.
    fwds : list of mne.Forward
        Forward models for each subject.
    evokeds : list of mne.Evoked
        Averaged sensor responses for each subject.
    noise_covs : list of mne.Covariance
        Noise covariance matrices for each subject.
    target : float
        Target count of average active source points.
    overwrite : Bool
        Whether or not overwrite existing source estimates.
    info : str, optional
        Additional info string to append to the beginning of a log file entry.
        Default is ''.
    suffix : str, optional
        Suffix to append to the end of the output filename, by default None.
    solver_kwargs : kwargs
        Additional parameters to be passed on to the underlying solver.

    Returns
    ----------
    None.
    '''

    # Check that stcs for all stimuli have been calculated and saved
    missing = False
    for subject in subjects:
        fname_stc = get_fname(subject, 'stc', src_spacing = src_spacing,
                              stc_method = stc_method, task = task, stim=stim,
                              suffix = suffix)
        fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
        if not os.path.isfile(fpath_stc + '-lh.stc'):
            print(fpath_stc + " Doesn't exist")
            missing = True
            break
    
    if missing == False and overwrite == False:
        return

    if stc_method == "remtw":
        # Set default values if they are not set in function call
        if 'concomitant' not in solver_kwargs:
            solver_kwargs['concomitant'] = False
        if 'epsilon' not in solver_kwargs:
            solver_kwargs['epsilon'] = 5. / fwds[0]['sol']['data'].shape[-1]
        if 'gamma' not in solver_kwargs:
            solver_kwargs['gamma'] = 1

        # Find 0.5 * alpha_max, where alpha_max spreads activation everywhere
        if 'alpha' not in solver_kwargs or solver_kwargs['alpha'] == None:
            print('Finding optimal alpha for ' + stim)
            _, alpha = reMTW_find_param(fwds, evokeds, noise_covs, stim,
                                        project_dir, copy.deepcopy(solver_kwargs),
                                        param = 'alpha', info = info,
                                        suffix = suffix)
            solver_kwargs['alpha'] = alpha
        
        # Find beta which produces exactly <target> active source points
        if 'beta' not in solver_kwargs or solver_kwargs['beta'] == None:
            print('Finding optimal beta for ' + stim)
            stcs, _ = reMTW_find_param(fwds, evokeds, noise_covs, stim,
                                       project_dir, solver_kwargs,
                                       target = target, param = 'beta',
                                       info = info, suffix = suffix)
        
        # Everything has been set beforehand, just run the inversion
        else:
            # Try 5 times to get a solid estimate, if fails 5 times return
            # without writing stcs to file.
            for i in range(5):
                try:
                    stcs, _ = reMTW_wrapper(fwds, evokeds, noise_covs,
                                            solver_kwargs)
                    break
                except ValueError as e:
                    print("Beta=" + str(solver_kwargs['beta'])
                          + " caused an error (skipping):")
                    print(e)
                    print('Reducing beta by 0.01...\n')
                    solver_kwargs['beta'] -= 0.01
                    if i == 4:
                        print('Failed to acquire stcs. No stcs saved.')
                        return

    # Save the returned source time course estimates to disk
    print(stcs)
    for i, stc in enumerate(stcs):
        fname_stc = get_fname(subjects[i], 'stc', src_spacing = src_spacing,
                              stc_method = stc_method, task = task, stim = stim,
                              suffix = suffix)
        fpath_stc = os.path.join(project_dir, 'Data', 'stc', fname_stc)
        print(fpath_stc)
        stc.save(fpath_stc)
