#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions used exclusively in the reMTW pipeline.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import os
from datetime import datetime, timedelta
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from groupmne import compute_group_inverse

from .utils import multi_sort

def reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs):
    '''
    Wraps groupmne.compute_group_inverse for timing & calculating
    average active source points between all subjects.

    Parameters
    ----------
    fwds : list of mne.Forward
        Forward models of each subject that have been preprocessed with
        groupmne.prepare_forwards().
    evokeds : list of mne.Evoked
        Evoked responses for each subject.
    noise_covs : list of mne.Covariance
        Covariance matrices for each subject.
    solver_kwargs : dict
        Additional parameters to pass on to the reMTW solver.

    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates create by reMTW.
    avg : float
        Number of average active source points over subjects.
    '''

    start = timer()
    stcs = compute_group_inverse(fwds, evokeds, noise_covs, method = 'remtw',
                                 spatiotemporal = False,
                                 n_jobs = 15, stable = True, gpu = True,
                                 tol_ot = 1e-4, max_iter_ot = 20, tol = 1e-4,
                                 max_iter = 2000, positive = False,
                                 tol_reweighting = 1e-2,
                                 max_iter_reweighting = 100,
                                 ot_threshold = 1e-7,
                                 **solver_kwargs)
    stop = timer()
    print("Solved in " + str(timedelta(seconds = (stop - start))))

    # Calculate average active source points over all subjects
    avg = 0
    for stc in stcs:
        avg += np.count_nonzero(stc.data)
    avg = avg / len(stcs)

    return stcs, avg

def reMTW_param_plot(log, project_dir, param, stim, suffix = ''):
    '''
    Plot active source points vs. alpha or beta. Intermediate plots are
    generated by the pipeline each time a parameter search is run. Used
    also for Figure 2 when the reMTW pipeline is run using option '-hyperplot'.

    The plots are saved in <project_dir>/Data/plot/.

    Parameters
    ----------
    log : {<param> : [float], 'actives' : [float]}
        Average active source points corresponding to each
        given parameter value.
    project_dir : str
        Base directory of the project.
    param : str
        Plotted parameter, either 'alpha' or 'beta'.
    stim : str
        Stimulus name, e.g. 'sector21'.
    suffix : str, optional
        Optional suffix to include in the filename, by default ''.
    
    Returns
    ----------
    None.
    '''

    plt.ioff()
    plt.figure(figsize=(4, 2.5))
    plt.plot(log[param], log['actives'], '--bo')

    plt.yscale('log')
    plt.xlabel((r'$\mu$' if param == 'alpha' else r'$\lambda$'), fontsize = 15)
    plt.ylabel('AVG active sources', fontsize = 15)
    plt.tight_layout()
    plt.grid()

    # Set yticks to discrete values
    ax = plt.gca()
    plt.yticks([1, 5, 10, 50, 500, 1000, 2500, 5000])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())

    plt.savefig(os.path.join(project_dir, 'Data', 'plot',
                             f'{param}_{stim}_{suffix}.pdf'))
    plt.close()

def reMTW_save_params(project_dir, param_name, param_list, actives, sec_name,
                      sec_value, stim, suffix = '', info = ''):
    '''
    Save parameter log into a text file in <project_dir>/Data/plot/.

    Parameters
    ----------
    project_dir : str
        Base directory of the project.
    param_name : str
        Name of the logged parameter, e.g. 'alpha'.
    param_list : list of float
        Logged values of the parameter.
    actives : list of float
        Average active source points for given parameter value.
    sec_name : str
        Secondary parameter name which is kept static, e.g. 'beta'.
    sec_value : float
        Value of the secondary parameter.
    stim : str
        Name of the stimulus these values were acquired for.
    suffix : str, optional
        Optional suffix to include in the filename, by default ''.
    info : str, optional
        Additional info string to append to the beginning of a log file entry.
        By default ''.
    
    Returns
    -------
    None.
    '''

    fpath = os.path.join(project_dir, 'Data', 'plot', f'{stim}_{suffix}.txt')
    with open(fpath, 'a') as f:
        f.write(datetime.now().strftime("%D.%M.%Y %H:%M:%S") + f' {info}\n')
        f.write(f'{param_name} with {sec_name} = {sec_value}:\n')
        f.write(', '.join([str(value) for value in param_list]) + '\n')
        f.write('Active source points with given parameters:\n')
        f.write(', '.join([str(value) for value in actives]) + '\n')
        f.write('\n-----\n\n')

def reMTW_search_step(current, log, history, param):
    '''
    A modified binary search step. Outputs the next parameter to try based
    on the categorization of the last three values as either "big" or "small"
    reflecting their relationship to set target.
    
    Big > target and small < target.

    Parameters
    ----------
    current : float
        Current value of the stepped parameter.
    log : list of float
        Past values of the stepped parameter.
    history : list of str
        Last three evaluations of the result, either 'big' or 'small'.
    param : str
        Parameter incremented, e.g. 'alpha'.

    Returns
    -------
    step : float
        Next parameter value to be tested.
    '''

    if history[0] == history[1] and history[1] == history[2]:
        # Strolling on flat land, take a jump to get near the gradient faster
        if history[2] == 'small':
            if param == 'alpha':
                return current * 5
            if param == 'beta':
                return current + 0.1
        elif history[2] == 'big':
            if param == 'alpha':
                return current * 0.2
            if param == 'beta':
                # With small beta values the avg active sources rises again
                # and the computation time skyrockets
                if current >= 0.2:
                    return current - 0.1
                else:
                    return current * 1.5
    elif history[1] != history[2]:
        # Moved over optimum -> search the midpoint of these points
        return (log[-1] + log[-2]) / 2
    elif history[0] != history[1]:
        # Optimum wasn't between the indices [-1, -2] -> it's between [-1, -3]
        return (log[-1] + log[-3]) / 2

def find_aMax(alphas, actives):
    '''
    Find the value of A_max based on alpha values and average active source
    points for each alpha value.

    Parameters
    ----------
    alphas : list of float
        The alpha values corresponding to actives
    actives : list of float
        The average active source points corresponding to alphas

    Returns
    -------
    aMax : float
        The value of Alpha_max.
    '''

    aMax_idx = 0
    max_diff = 0
    for i in range(len(alphas) - 1):
        if actives[i + 1] - actives[i] > max_diff:
            aMax_idx = i
            max_diff = actives[i + 1] - actives[i]
    aMax = alphas[aMax_idx]

    return aMax

def reMTW_find_param(fwds, evokeds, noise_covs, stim, project_dir,
                     solver_kwargs, target = 50, param = 'alpha', info = '',
                     suffix = ''):
    '''
    Find a good value for given <param>. Alpha is found by testing different
    values until the estimate cover the whole cortex (alpha_max) and then
    1/2 * alpha_max is returned as a "safe" heuristic. Beta is chosen as the
    value with average active sources as close to <target> as possible.

    Parameters
    ----------
    fwds : list of mne.Forward
        Forward models for each subject, preprocessed with
        groupmne.prepare_forwards().
    evokeds : list of mne.Evoked
        Sensor responses to the stimulus, one per subject.
    noise_covs : list of mne.Covariance
        Noise covariance matrices for each subject.
    stim : str
        Stimulus which is analyzed here, e.g. 'sector22'.
    project_dir : str
        Base directory of the project.
    solver_kwargs : dict
        Additional parameters sent to the solver.
    target : int, optional
        How many active source points to aim for, give only for beta search.
        By default 50.
    param : str, optional
        Which parameter to search for, either 'alpha' or 'beta'.
        By default 'alpha'.
    info : str, optinal
        Additional info string to append to the beginning of a log file entry.
        By default ''.
    suffix : str, optional
        Optional suffix to include in the filename, by default ''.

    Returns
    -------
    stcs_ : mne.SourceEstimate
        Source estimate associated with the returned parameter value.
    param_value : float
        Optimal alpha or beta with given function parameters.
    '''

    log = {param : [], 'actives' : [], 'stcs' : []}
    secondary = 'beta' if param == 'alpha' else 'alpha'
    
    # Set baseline parameters
    if param == 'alpha':
        solver_kwargs['alpha'] = 1
    if solver_kwargs['concomitant'] == False:
        solver_kwargs['beta'] = 0.3 if param == 'alpha' else 0.4
    else:
        solver_kwargs['beta'] = 0.2

    history = []
    avg = 0
    iter = 1
    max_iter = 15

    while True:
        if param == 'beta':
            # Close enough to target or max_iter exhausted. Beta needs more
            # iterations to converge compared to alpha.
            if 0.8 * target < avg < 1.2 * target or iter > max_iter:
                break

            # Adjust beta if a value is recomputed
            if solver_kwargs['beta'] in log['beta']:
                solver_kwargs['beta'] -= 0.1 * solver_kwargs['beta']
                iter += 1
                continue
        else:
            # A good enough alpha_max can be found with 7 iterations.
            if iter > 7:
                break
        
        # Compute new source estimates and average active source points
        try:
            print(f'Solving for {param} = {solver_kwargs[param]} with '
                  + f'{secondary} = {solver_kwargs[secondary]}')

            stcs, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print(f'Got {avg} active sources with {param} = '
                  + f'{solver_kwargs[param]}')
        except ValueError as e:
            print(f'{param} = {solver_kwargs[param]} caused an error (skipping):')
            print(e)
            if param == 'alpha':
                solver_kwargs[param] *= 2
            else:
                solver_kwargs[param] -= 0.05 * solver_kwargs[param]
            continue

        # Update log and history
        if len(history) == 0:
            history = ['small' if avg < target else 'big'] * 3
        else:
            history.append('small' if avg < target else 'big')
            history.pop(0)

        log[param].append(solver_kwargs[param])
        log['actives'].append(avg)
        log['stcs'].append(stcs)
        
        # Get next value to try for the parameter
        solver_kwargs[param] = reMTW_search_step(solver_kwargs[param],
                                                 log[param], history, param)
        if solver_kwargs[param] < 0:
            solver_kwargs[param] = abs(solver_kwargs[param])
       
        print(log[param], log['actives'])
        iter += 1

    log[param], [log['actives'], log['stcs']] = multi_sort(log[param],
                                                           [log['actives'],
                                                           log['stcs']])
    
    if param == 'alpha':
        # Find the elbow = the highest gradient as alpha_max
        # Good heuristic for alpha is 0.5 * aMax
        aMax = find_aMax(log[param], log['actives'])
        stcs_ = log['stcs'][log[param].index(aMax)]
        print(f'Got aMax = {aMax}')
        param_value = 0.5 * aMax
    else:
        # Select the beta with avg closest to the target
        beta_idx = np.argmin(np.abs(np.array(log['actives']) - target))
        param_value = log[param][beta_idx]
        stcs_ = log['stcs'][beta_idx]
        print(f'Got beta_ = {param_value}')

    # Print the result, save log in a file and plot active points vs alphas
    reMTW_param_plot(log, project_dir, param, stim, suffix = suffix)
    reMTW_save_params(project_dir, param, log[param], log['actives'],
                      secondary, solver_kwargs[secondary], stim, info = info,
                      suffix = suffix)

    return stcs_, param_value

def reMTW_hyper_plot(fwds, evokeds, noise_covs, stim, project_dir,
                     concomitant = False, param = 'alpha', secondary = 0.3):
    '''
    Plot the relationship between alpha and average active sources at multiple
    points. Used to output plots in Figure 2.

    Parameters
    ----------
    fwds : list of mne.Forward
        Forward models for each subject, preprocessed with
        groupmne.prepare_forwards().
    evokeds : list of mne.Evoked
        Sensor responses to the stimulus, one per subject.
    noise_covs : list of mne.Covariance
        Noise covariance matrices for each subject.
    stim : str
        Stimulus which is analyzed here, e.g. 'sector22'.
    project_dir : str
        Base directory of the project.
    concomitant : bool, optional
        Whether to use concomitant noise estimation or not, by default False
    param : str, optional
        Which parameter to plot, either 'alpha' or 'beta', by default 'alpha'.
    secondary : float, optional
        Static value for the secondary parameter, if param == 'alpha',
        secondary means beta and vice versa. By default 0.3
    '''

    log = {param : [], 'actives' : []}
    secondary_param = 'alpha' if param == 'beta' else 'beta'

    solver_kwargs = {param : (1 if param == 'alpha' else 0.3),
                     secondary_param : secondary}
    solver_kwargs['epsilon'] = 5. / fwds[0]['sol']['data'].shape[-1]
    solver_kwargs['gamma'] = 1
    solver_kwargs['concomitant'] = concomitant

    if param == 'alpha':
        params = np.linspace(1, 25, 13)
    else:
        params = np.linspace(0.2, 0.9, 15)

    # Find the number of average sources with each primary hyperparameter value
    for p in params:
        solver_kwargs[param] = p
        try:
            print(f'Solving for {param} = {solver_kwargs[param]}')
            _, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print(f'Got {avg} active sources with {param} = {solver_kwargs[param]}')
        except ValueError as e:
            print(f'{param} = {solver_kwargs[param]} caused an error (skipping):')
            print(e)
            solver_kwargs[param] += (1 if param == 'alpha' else 0.05)
            continue

        log[param].append(solver_kwargs[param])
        log['actives'].append(avg)

    log['actives'] = [avg for _, avg in sorted(zip(log[param], log['actives']))]
    log[param].sort()

    # Print the result, save log in a file and plot active points vs params
    reMTW_param_plot(log, project_dir, param, stim, suffix = 'hyperplot')
    reMTW_save_params(project_dir, param, log[param], log['actives'],
                      secondary_param, solver_kwargs[secondary_param], stim)
