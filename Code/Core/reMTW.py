#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions used exclusively in the reMTW pipeline.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from groupmne import compute_group_inverse

def reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs):
    '''
    Wraps groupmne.compute_group_inverse for parallel processing & calculates
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

    # Calculate inverse solutions with groupmne's reMTW
    stcs = compute_group_inverse(fwds, evokeds, noise_covs, method = 'remtw',
                                 spatiotemporal = False,
                                 n_jobs = 15, stable=True, gpu = True,
                                 tol_ot=1e-4, max_iter_ot=20, tol=1e-4,
                                 max_iter=2000, positive=False, tol_reweighting = 1e-2,
                                 max_iter_reweighting = 100, **solver_kwargs)

    # Calculate average active source points over all subjects
    avg = 0
    for stc in stcs:
        avg += np.count_nonzero(stc.data)
    avg = avg / len(stcs)

    return (stcs, avg)

def reMTW_param_plot(log, project_dir, param, stim, fname_id = ""):
    '''
    Plot active source points vs. alpha or beta.

    Parameters
    ----------
    log : list of float
        Average active source points for each given parameter value.
    project_dir : str
        Base directory of the project.
    param : str
        Plotted parameter, e.g. 'alpha'.
    stim : str
        Stimulus name, e.g. 'sector21'.
    fname_id : str
        Additional string to append to the filename. Default is "".
    
    Returns
    ----------
    None.
    '''

    plt.ioff()
    plt.plot(log[param], log['actives'], '--bo')
    plt.yscale('log')
    plt.xlabel(param)
    plt.ylabel('Average active sources')
    plt.title(stim + ' ' + param)

    # Set yticks to discrete values
    ax = plt.gca()
    plt.yticks([1, 5, 10, 50, 500, 1000, 2500, 5000])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())

    # Save
    plt.savefig(project_dir + 'Data/plot/' + param + '_' + stim + fname_id + '.png')
    plt.close()

def reMTW_save_params(project_dir, param_name, param_list, actives, sec_name,
                      sec_value, stim):
    '''Save parameter log into a text file.

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
    
    Returns
    -------
    None.
    '''

    with open(project_dir + 'Data/plot/' + stim + '.txt', 'a') as f:
        f.write(datetime.now().strftime("%D.%M.%Y %H:%M:%S") + '\n')
        f.write(param_name + ' with ' + sec_name + '=' + str(sec_value) + ':\n')
        f.write(', '.join([str(value) for value in param_list]) + '\n')
        f.write('Active source points with given parameters:\n')
        f.write(', '.join([str(value) for value in actives]) + '\n')
        if param_name == 'beta':
            f.write('\n-----\n\n')

def reMTW_search_step(current, log, history, param):
    '''
    A modified binary search step.

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
        # Strolling a plateau, take a jump to get near the gradient faster
        if history[2] == 'small':
            if param == 'alpha':
                return current * 5
            if param == 'beta':
                return current + 0.1
        elif history[2] == 'big':
            if param == 'alpha':
                return current * 0.2
            if param == 'beta':
                return current - 0.1
    elif history[1] != history[2]:
        # Moved over optimum -> search the midpoint of these points
        return (log[-1] + log[-2]) / 2
    elif history[0] != history[1]:
        # Optimum was not between the point index [0, -1] -> it's between [0, -2]
        return (log[-1] + log[-3]) / 2

def reMTW_find_alpha(fwds, evokeds, noise_covs, stim, project_dir, solver_kwargs):
    '''
    Find alpha_max where the source estimate covers the whole cortex, a heuristic
    for optimal alpha is alpha_max / 2.

    Parameters
    ----------
    fwds : list of mne.Forward
        Forward models for each subject, preprocessed with groupmne.prepare_forwards().
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

    Returns
    -------
    alpha : float
        0.5 * alpha_max, a safe and good alpha value for this stimulus
    '''

    log = dict(alphas = [], actives = [])
    
    # Set baseline parameters
    solver_kwargs['alpha'] = 1
    if solver_kwargs['concomitant'] == False:
        solver_kwargs['beta'] = 0.3
    else:
        solver_kwargs['beta'] = 0.7

    history = []
    # Find alpha with 7 iterations
    for i in range(7):
        try:
            print("Solving for alpha=" + str(solver_kwargs['alpha']))
            _, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print("Got " + str(avg) + " active sources with alpha=" + str(solver_kwargs['alpha']))
        except ValueError as e:
            print("Alpha=" + str(solver_kwargs['alpha']) + " caused an error (skipping):")
            print(e)
            solver_kwargs['alpha'] *= 2
            continue

        if len(history) == 0:
            history = ['small' if avg < 50 else 'big'] * 3
            log['alphas'] += [solver_kwargs['alpha']]
            log['actives'] += [avg]
        else:
            history.append('small' if avg < 50 else 'big')
            history.pop(0)
            log['alphas'].append(solver_kwargs['alpha'])
            log['actives'].append(avg)
        
        solver_kwargs['alpha'] = reMTW_search_step(solver_kwargs['alpha'], log['alphas'],
                                                   history, 'alpha')
        print(log)

    # Find the elbow = the highest gradient as alpha_max
    log['actives'] = [avg for alpha, avg in sorted(zip(log['alphas'], log['actives']))]
    log['alphas'].sort()

    aMax_idx = 0
    max_diff = 0
    for i in range(len(log['alphas']) - 1):
        if log['actives'][i + 1] - log['actives'][i] > max_diff:
            aMax_idx = i
            max_diff = log['actives'][i + 1] - log['actives'][i]
    aMax = log['alphas'][aMax_idx]

    # Print the result, save log in a file and plot active points vs alphas
    print("Got aMax=" + str(aMax))
    reMTW_param_plot(log, project_dir, 'alphas', stim)
    reMTW_save_params(project_dir, 'alpha', log['alphas'], log['actives'], 'beta', solver_kwargs['beta'], stim)

    # Good heuristic for alpha is 0.5 * aMax
    return 0.5 * aMax

def reMTW_find_beta(fwds, evokeds, noise_covs, stim, project_dir, target,
                    solver_kwargs):
    '''
    Find beta where number of active source points is as close to the target
    as possible.

    Parameters
    ----------
    fwds : list of mne.Forward
        Forward models for each subject, preprocessed with groupmne.prepare_forwards().
    evokeds : list of mne.Evoked
        Sensor responses to the stimulus, one per subject.
    noise_covs : list of mne.Covariance
        Noise covariance matrices for each subject.
    stim : str
        Stimulus which is analyzed here, e.g. 'sector22'.
    project_dir : str
        Base directory of the project.
    target : float
        Target count of active source points.
    solver_kwargs : dict
        Additional parameters sent to the solver.

    Returns
    -------
    stcs_ : list of mne.SourceEstimate
        Source estimates attained with beta_.
    beta_ : float
        Optimized value of reMTW beta parameter.
    '''

    log = dict(betas = [], actives = [], stcs = [])
    
    # Set baseline parameters
    solver_kwargs['beta'] = 0.4
    history = []
    avg = 0
    iter = 0
    max_iter = 20

    # Find beta with a maximum of max_iter iterations
    while (avg > 1.2 * target or avg < 0.8 * target) and iter < max_iter:
        # Check if the solver is repeating steps:
        if solver_kwargs['beta'] in log['betas']:
            solver_kwargs['beta'] -= 0.1 * solver_kwargs['beta']
            continue

        # Solve for given beta, ValueError is quite common occurence
        iter += 1
        try:
            print("Solving for beta=" + str(solver_kwargs['beta']))
            stcs, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print("Got " + str(avg) + " active sources with beta=" + str(solver_kwargs['beta']))
        except ValueError as e:
            print("Beta=" + str(solver_kwargs['beta']) + " caused an error (skipping):")
            print(e)
            solver_kwargs['beta'] += 0.1 * solver_kwargs['beta']
            continue

        # Update log and history
        if len(history) == 0:
            history = ['small' if avg < target else 'big'] * 3
            log['betas'] += [solver_kwargs['beta']]
            log['actives'] += [avg]
            log['stcs'] += [stcs]
        else:
            history.append('small' if avg < target else 'big')
            history.pop(0)
            log['betas'].append(solver_kwargs['beta'])
            log['actives'].append(avg)
            log['stcs'] += [stcs]

        # Find the next beta value to test
        solver_kwargs['beta'] = reMTW_search_step(solver_kwargs['beta'], log['betas'],
                                                  history, 'beta')
        print(log['betas'], log['actives'])

    # Select the beta with avg closest to the target
    beta_idx = np.argmin(np.abs(np.array(log['actives']) - target))
    beta_ = log['betas'][beta_idx]
    stcs_ = log['stcs'][beta_idx]

    # Plot avg vs beta
    log['actives'] = [avg for alpha, avg in sorted(zip(log['betas'], log['actives']))]
    log['betas'].sort()
    reMTW_param_plot(log, project_dir, 'betas', stim)
    reMTW_save_params(project_dir, 'beta', log['betas'], log['actives'], 'alpha', solver_kwargs['alpha'], stim)

    print("Got beta_=" + str(beta_))
    return (stcs_, beta_)

def reMTW_tenplot_a(fwds, evokeds, noise_covs, stim, project_dir):

    print("Solving for stimulus " + stim)
    stim_idx = int("".join([i for i in stim if i in "1234567890"])) - 1
    evokeds = [ev.crop(0.08,0.08) for ev in evokeds[stim_idx]]
    print("Stimulus ID (sector - 1): " + str(stim_idx))

    log = dict(alphas = [], actives = [])
    
    solver_kwargs = dict(beta=0.3, alpha=1)
    solver_kwargs['epsilon'] = 5. / fwds[0]['sol']['data'].shape[-1]
    solver_kwargs['gamma'] = 1
    #solver_kwargs['concomitant'] = False

    alphas = np.linspace(0,25,11)
    alphas[0] = 1

    # Test alpha at 11 points
    for a in alphas:
        solver_kwargs['alpha'] = a
        try:
            print("Solving for alpha=" + str(solver_kwargs['alpha']))
            _, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print("Got " + str(avg) + " active sources with alpha=" + str(solver_kwargs['alpha']))
        except ValueError as e:
            print("Alpha=" + str(solver_kwargs['alpha']) + " caused an error (skipping):")
            print(e)
            solver_kwargs['alpha'] += 1
            continue

        log['alphas'] += [solver_kwargs['alpha']]
        log['actives'] += [avg]

    log['actives'] = [avg for alpha, avg in sorted(zip(log['alphas'], log['actives']))]
    log['alphas'].sort()

    # Print the result, save log in a file and plot active points vs alphas
    reMTW_param_plot(log, project_dir, 'alphas', stim, 'tenpoint')
    reMTW_save_params(project_dir, 'alpha', log['alphas'], log['actives'], 'beta', solver_kwargs['beta'], stim)

def reMTW_tenplot_b(fwds, evokeds, noise_covs, stim, project_dir, alpha = 7.5):

    print("Solving for stimulus " + stim)
    stim_idx = int("".join([i for i in stim if i in "1234567890"])) - 1
    evokeds = [ev.crop(0.08,0.08) for ev in evokeds[stim_idx]]
    print("Stimulus ID (sector - 1): " + str(stim_idx))

    log = dict(betas = [], actives = [])
    
    solver_kwargs = dict(beta=0.3, alpha=alpha)
    solver_kwargs['epsilon'] = 5. / fwds[0]['sol']['data'].shape[-1]
    solver_kwargs['gamma'] = 1
    #solver_kwargs['concomitant'] = False
    betas = np.linspace(0.2,0.9,11)

    # Test betas at 11 points
    for b in betas:
        solver_kwargs['beta'] = b
        try:
            print("Solving for beta=" + str(solver_kwargs['beta']))
            _, avg = reMTW_wrapper(fwds, evokeds, noise_covs, solver_kwargs)
            print("Got " + str(avg) + " active sources with beta=" + str(solver_kwargs['beta']))
        except ValueError as e:
            print("beta=" + str(solver_kwargs['beta']) + " caused an error (skipping):")
            print(e)
            solver_kwargs['beta'] += 0.05
            continue

        log['betas'] += [solver_kwargs['beta']]
        log['actives'] += [avg]

    log['actives'] = [avg for beta, avg in sorted(zip(log['betas'], log['actives']))]
    log['betas'].sort()

    # Print the result, save log in a file and plot active points vs alphas
    reMTW_param_plot(log, project_dir, 'betas', stim, 'tenpoint')
    reMTW_save_params(project_dir, 'beta', log['betas'], log['actives'], 'alpha', solver_kwargs['alpha'], stim)