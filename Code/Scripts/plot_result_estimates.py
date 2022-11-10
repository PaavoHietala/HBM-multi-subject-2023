#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:31:07 2021

Produces all 9 source estimates for Figure 4: Source estimates.
First, 3 eLORETA images are plotted followed by 3 reMTW and 3 reMTW & AVG plots.

Plots are saved to <project_dir>/Data/plot/<filename of stc>.png

@author: hietalp2
"""

import os.path as op

import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import (make_axes_locatable)

import sys
import mne

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core import utils

# Script settings --------------------------------------------------------------

# Root data directory for each source estimate, list of str

project_dirs = ['/m/nbe/scratch/megci/MFinverse/Classic/'] * 3 \
               + ['/m/nbe/scratch/megci/MFinverse/reMTW/'] * 6

# Subjects' MRI location (FreeSurfer dir), str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'

# List of subjects, indices within all following lists must match

subjects = ['MEGCI_S1', 'fsaverage', 'fsaverage'] + ['MEGCI_S1'] * 3 \
            + ['fsaverage'] * 3

# List of stimuli

stims = ['sector16'] * 9

# List of inverse solution methods used for each stc

methods = ['eLORETA'] * 3 + ['remtw'] * 6

# Suffixes of stcs, can be None for individual stcs

suffixes = [None, '10subjects', '20subjects'] \
           + ['1subjects', '10subjects', '20subjects'] * 2

# Types of each stc. 'stc' (individual), 'stc_m' (individual on fsaverage mesh)
# or 'avg' (averaged data on fsaverage mesh)

stc_types = ['stc_m', 'avg', 'avg'] + ['stc_m'] * 3 + ['avg'] * 3
            
# Target source space of the morphing in stc_m and avg, usually 'fsaverage'

src_tos = ['fsaverage', None, None] + ['fsaverage'] * 3 + [None, None, None]

# Get peak for selected time only (used only if the stc has multiple timepoints)

time = 0.081

# Plot colorbar from 0 to max (abs) or from -max to +max (bi), abs to separate
# file (sep), bi to separate file (bisep) or None

cbars = ['sep', None, None] + ['bisep'] * 3 + ['sep'] * 3

# Overwrite existing files

overwrite = True

# Custom clims, mainly for eLORETA to set all 3 to same limits

abs_max = 0
for stim, suffix in zip(stims[:3], suffixes[:3]):
    stc = mne.read_source_estimate(project_dirs[0]
                                   + 'Data/avg/fsaverage-ico4-eLORETA-f-'
                                   + stim + ('-' + suffix if suffix else ''))
    if np.max(abs(stc.data)) > abs_max:
        abs_max = np.max(abs(stc.data))

clims = [{'kind' : 'value', 'lims' : [0, 0.5 * abs_max, abs_max]}] * 3 + [None] * 6

# Run the script for all data --------------------------------------------------

mne.set_config('SUBJECTS_DIR', subjects_dir)

def plot_result_estimates(subject, stim, method, project_dir, suffix = None,
                          stc_type = 'stc', src_to = 'fsaverage',
                          cbar_type = 'abs', overwrite = False,
                          clim = None):
    '''
    Monster of a function to plot source estimates with or without colorbars or
    source estimates & colorbars in separate files.

    Parameters
    ----------
    subject : str
        Subject for which to draw the plot, e.g. 'fsaverage'.
    stim : str
        Stimulus name, e.g. 'sector1'.
    method : str
        Inverse solution method, e.g. 'eLORETA'.
    project_dir : str
        Base directory of the project with Data subfolder.
    suffix : str
        Suffix to append to the end of the output filename, by default None
    stc_type : str, optional
        Stc type, either 'stc', 'stc_m' or 'avg', by default 'stc'
    src_to : str, optional
        Mesh to which the stc was morphed beforehand (what mesh to actually draw
        the plot on IF the stc has been morphed), None if the stc has not been
        morphed. The default is 'fsaverage'.
    cbar_type : str, optional
        Colorbar type, either 'abs' for positive only or 'bi' for neg-0-pos,
        by default 'abs'. 'sep' and 'bisep' for separate CB files respectively.
    overwrite : bool, optional
        Overwrite existing files switch. The default is False.
    clim : dict, optional
        Clims for the color bar, by default None for automatic clims.
    
    Returns
    -------
    None.
    '''

    # Load source estimate
    fname_stc = '-'.join(filter(None, [subject, 'ico4', method, src_to, 'f',
                                       stim, suffix, 'lh.stc']))
    fpath_stc = op.join(project_dir, 'Data', stc_type, fname_stc)

    fpath_out = op.join(project_dir, 'Data', 'plot', fname_stc + '.png')

    if not overwrite and op.isfile(fpath_out):
        print('File ' + fpath_stc + ' exists and overwrite = False, skipping\n')
        return

    print('Reading', fpath_stc)
    stc = mne.read_source_estimate(fpath_stc)

    # Set one point to very low value in remtw to avoid an all-blue plot caused
    # by an all-zero estimate
    if method == 'remtw':
        if np.count_nonzero(stc.lh_data) == 0:
            stc.lh_data[0] = 1e-20
        if np.count_nonzero(stc.rh_data) == 0:
            stc.rh_data[0] = 1e-20
        
    # Plot the STC with V1 borders:
    brain = stc.plot(hemi='split', size=(1500,600),
                     subject = (src_to if stc_type == 'stc_m' else subject),
                     initial_time = (time if stc.data.shape[1] > 1 else None),
                     background = 'w', colorbar = False,
                     time_viewer = False, show_traces = False,
                     clim = (clim if clim else 'auto'))
    
    # Add peak foci
    bilaterals = [f'sector{n}' for n in [3, 7, 11, 15, 19, 23]]
    peaks, peak_hemis = utils.find_peaks(project_dir, 'ico4', method, 'f',
                                         [stim], bilaterals, suffix, stc = stc,
                                         time = time)
    print(peaks, peak_hemis, stim, bilaterals)

    for peak, hemi in zip(peaks, peak_hemis):
        brain.add_foci(peak, coords_as_verts = True, scale_factor = 1,
                       color = 'lime', hemi = hemi)
    
    # Set colorbar limits and type
    if cbar_type == 'sep' and clim != None:
        maxi = clim['lims'][2]
    else:
        if stc.data.shape[1] > 1:
            maxi = abs(stc.data[:, 129]).max()
        else:
            maxi = abs(stc.data).max()
    
    if clim == None and cbar_type:
        if cbar_type in ['abs', 'sep']:
            clim = dict(kind = 'value', lims = [0, 0.5 * maxi, maxi])
        elif cbar_type in ['bi', 'bisep']:
            clim = dict(kind='value', pos_lims=[0, 0.5 * maxi, maxi])
    
    # Add V1 border
    for hemi in ['lh', 'rh']:
        v1 = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + 'fsaverage/label/' + hemi + '.V1_exvivo.label',
                            'fsaverage')
        brain.add_label(v1, borders = 2, color = 'lime')

    # Rotate the brain, take a snapshot and remove all white rows / columns
    brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 400, col = 0)
    brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 400, col = 1)
    screenshot = brain.screenshot()
    cropped_screenshot = utils.crop_whitespace(screenshot)
    brain.close()

    # Tweak the figure style
    plt.rcParams.update({
        'font.size' : 22,
        'grid.color': '0.75',
        'grid.linestyle': ':',
    })

    # Create new fig with subplots to get axes easily
    fig, axes = plt.subplots(num = fname_stc, figsize = (10, 7.))

    # now add the brain to the axes
    axes.imshow(cropped_screenshot)
    axes.axis('off')

    # Save source estimates if colorbar is rendered separately
    if cbar_type in [None, 'sep', 'bisep']:
        plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0)
        if cbar_type == None:
            return
        else:
            fpath_out = fpath_out[:-4] + '-colorbar.png'
            fig, axes = plt.subplots(num = fname_stc + '-cb', figsize = (7, 7.))
            axes.axis('off')

    # add a horizontal colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('bottom', size = '5%', pad = 0.2)

    # Redefine the color bar ticks, if in scientific format add the exponent
    # to the label
    print(clim, cbar_type, maxi)
    try:
        base, exponent = str(maxi).split('e')
        base = str(round(float(base), 2))
        cbar = mne.viz.plot_brain_colorbar(cax, clim,
                                           orientation = 'horizontal',
                                           label = r'Activation (Am) '
                                           + r'$\cdot 10^{'
                                           + exponent + r'}$')
        if cbar_type in ['bi', 'bisep']:
            cbar.set_ticks([-maxi, 0, maxi])
            cbar.set_ticklabels(["%.2f" % -float(base), "%.2f" % 0.00,
                                 "%.2f" % float(base)])
        else:
            cbar.set_ticks([0, maxi / 2, maxi])
            cbar.set_ticklabels(["%.2f" % 0.00, "%.2f" % (float(base) / 2),
                                 "%.2f" % float(base)])
            
    # Split fails = not exponential. Use plain floats as ticks instead
    except ValueError:
        print('Splitting exponent failed')
        cbar = mne.viz.plot_brain_colorbar(cax, clim,
                                           orientation = 'horizontal',
                                           label = 'Activation (Am)')
        if cbar_type in ['bi', 'bisep']:
            cbar.set_ticks([round(-maxi, 2), 0, round(maxi, 2)]) 
            cbar.set_ticklabels(["%.2f" % -maxi, "%.2f" % 0.00, "%.2f" % maxi])
        else:    
            cbar.set_ticks([0, round(maxi / 2, 2), round(maxi, 2)]) 
            cbar.set_ticklabels(["%.2f" % 0.00, "%.2f" % (maxi / 2), "%.2f" % maxi])
    
    cbar.outline.set_visible(True)

    # tweak margins and spacing
    fig.subplots_adjust(left = 0.15, right = 0.9, bottom = 0.01, top = 0.9,
                        wspace = 0.1, hspace = 0.5)

    # Save image        
    if cbar_type in ['sep', 'bisep']:
        plt.tight_layout()

        # Save colorbar to buffer
        buf = BytesIO()
        fig.savefig(buf, dpi = 500)
        buf.seek(0)
        cb = Image.open(buf)

        # Crop whitespace around the colorbar
        crop_sc = utils.crop_whitespace(np.asarray(cb), borders_only = True)
        cb = Image.fromarray(crop_sc)

        cb.save(fpath_out)
    else:
        plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0)

for i in range(len(subjects)):
    plot_result_estimates(subjects[i], stims[i], methods[i], project_dirs[i],
                          suffixes[i], stc_types[i], src_tos[i], cbars[i],
                          overwrite = overwrite, clim = clims[i])