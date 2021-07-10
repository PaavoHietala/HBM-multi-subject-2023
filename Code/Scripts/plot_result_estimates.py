#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:31:07 2021

Plot source estimates for section 4.1. of the Thesis.

Adapted from https://mne.tools/stable/auto_examples/visualization/publication_figure.html

@author: hietalp2
"""

import os.path as op

import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)
import sys
sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core import utils

import mne

#
# Script settings --------------------------------------------------------------
#
# Produces all 18 figures for thesis section 4.1. Source estimates.
# First, 6 eLORETA images are plotted followed by 6 reMTW & AVG and 6 plain
# reMTW plots.
#
# Plots are saved to <project_dir>/Data/plot/<filename of stc>
#

# Root data directory of the project, str

project_dirs = ['/m/nbe/scratch/megci/MFinverse/Classic/'] * 6 \
               + ['/m/nbe/scratch/megci/MFinverse/reMTW/'] * 12

# Subjects' MRI location, str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# List of subjects, indices within all following lists must match

subjects = ['MEGCI_S1', 'fsaverage', 'fsaverage'] * 2 \
           + ['fsaverage', 'fsaverage', 'fsaverage'] * 2 \
           + ['MEGCI_S1'] * 6

# List of stimuli

stims = (['sector16'] * 3 + ['sector15'] * 3) * 4

# List of methods used for each stc

methods = ['eLORETA'] * 6 + ['remtw'] * 12

# Suffixes of stcs, can be None for indiviudal stcs

suffixes = [None, '10subjects', '20subjects'] * 2 \
           + ['1subjects', '10subjects', '20subjects'] * 4

# Types of each stc. stc, stc_m or avg

stc_types = ['stc_m', 'avg', 'avg'] * 2 \
            + ['avg', 'avg', 'avg'] * 2 \
            + ['stc_m'] * 6

# Target source space of stc_m and avg, usually fsaverage

src_tos = ['fsaverage', None, None] * 2 \
          + [None, None, None] * 2 \
          + ['fsaverage'] * 6

# Get peak for selected time only

time = 0.081

# Plot colorbar from 0 to max (abs) or from -max to +max (bi), abs to separate
# file (sep) or None

cbars = [None] * 5 + ['sep'] + ['abs', 'abs', 'abs'] * 2 + ['bi'] * 6

# Overwrite existing files

overwrite = True

# Custom clims, mainly for eLORETA to set all 6 to same limits

abs_max = 0
for stim, suffix in zip(stims[:6], suffixes[:6]):
    stc = mne.read_source_estimate(project_dirs[0]
                                   + 'Data/avg/fsaverage-ico4-eLORETA-f-'
                                   + stim + ('-' + suffix if suffix else ''))
    if np.max(abs(stc.data)) > abs_max:
        abs_max = np.max(abs(stc.data)) * 0.75

clims = [{'kind' : 'value', 'lims' : [0, 0.5 * abs_max, abs_max]}] * 6 + [None] * 12

#
# Run the script for all data --------------------------------------------------
#

def plot_result_estimates(subject, stim, method, project_dir, suffix = None,
                          stc_type = 'stc', src_to = 'fsaverage',
                          cbar_type = 'abs', overwrite = False,
                          clim = None):
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

    # Set one point to very low value in remtw to avoid
    # all-blue plot caused by all-zero estimate
    if method == 'remtw':
        if np.count_nonzero(stc.lh_data) == 0:
            stc.lh_data[0] = 1e-20
        if np.count_nonzero(stc.rh_data) == 0:
            stc.rh_data[0] = 1e-20
        
    # Plot the STC with V1 borders, get the brain image, crop it:
    brain = stc.plot(hemi='split', size=(1500,600),
                     subject = ( src_to if stc_type == 'stc_m' else subject),
                     initial_time = (time if stc.data.shape[1] > 1 else None),
                     background = 'w', colorbar = False,
                     time_viewer = False, show_traces = False,
                     clim = (clim if clim else 'auto'))
    
    # Add peak foci
    bilaterals = ['sector3', 'sector7', 'sector11', 'sector15', 'sector19', 'sector23']
    peaks, peak_hemis = utils.find_peaks(project_dir, 'ico4', method, 'f',
                                         [stim], bilaterals,
                                         suffix, stc = stc, time = time)
    print(peaks, peak_hemis, stim, bilaterals)

    for peak, hemi in zip(peaks, peak_hemis):
        brain.add_foci(peak, coords_as_verts = True, scale_factor = 1,
                       color = 'lime', hemi = hemi)
    
    # Set colorbar limits and type
    if cbar_type == 'sep':
        maxi = clim['lims'][2]
    else:
        if stc.data.shape[1] > 1:
            maxi = abs(stc.data[:, 129]).max()
        else:
            maxi = abs(stc.data).max()
    
    if clim == None and cbar_type:
        if cbar_type == 'abs':
            clim = dict(kind = 'value', lims = [0, 0.5 * maxi, maxi])
        elif cbar_type == 'bi':
            clim = dict(kind='value', pos_lims=[0, 0.5 * maxi, maxi])
    
    for hemi in ['lh', 'rh']:
        v1 = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                            + 'fsaverage/label/' + hemi + '.V1_exvivo.label',
                            'fsaverage')
        brain.add_label(v1, borders = 2, color = 'lime')

    brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 400, col = 0)
    brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 400, col = 1)
    screenshot = brain.screenshot()
    brain.close()

    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    # Tweak the figure style
    plt.rcParams.update({
        'ytick.labelsize': 'large',
        'xtick.labelsize': 'large',
        'axes.labelsize': 'large',
        'axes.titlesize': 'large',
        'grid.color': '0.75',
        'grid.linestyle': ':',
    })

    # Create new fig with subplots to get axes easily
    fig, axes = plt.subplots(num = fname_stc, figsize = (4, 4.))

    # now add the brain to the axes
    axes.imshow(cropped_screenshot)
    axes.axis('off')

    if cbar_type == None or cbar_type == 'sep':
        plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0)
        if cbar_type == None:
            return
        else:
            fpath_out = fpath_out[:-4] + '-colorbar.png'
            fig, axes = plt.subplots(num = fname_stc + '-cb', figsize = (3, 4.))
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
        if cbar_type == 'bi':
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
        if cbar_type == 'bi':
            cbar.set_ticks([round(-maxi, 2), 0, round(maxi, 2)]) 
            cbar.set_ticklabels(["%.2f" % -maxi, "%.2f" % 0.00, "%.2f" % maxi])
        else:    
            cbar.set_ticks([0, round(maxi / 2, 2), round(maxi, 2)]) 
            cbar.set_ticklabels(["%.2f" % 0.00, "%.2f" % (maxi / 2), "%.2f" % maxi])
    # tweak margins and spacing
    fig.subplots_adjust(left = 0.15, right = 0.9, bottom = 0.01, top = 0.9,
                        wspace = 0.1, hspace = 0.5)

    # Save image        
    if cbar_type == 'sep':
        plt.tight_layout()
        # Save colorbar to buffer

        buf = BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        cb = Image.open(buf)
        cb.load()

        # Crop whitespace around the colorbar
        screenshot = np.asarray(cb)
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_indices = np.where(nonwhite_row == True)[0]
        nonwhite_row[nonwhite_indices[0] : nonwhite_indices[-1]] = True

        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        cb = Image.fromarray(cropped_screenshot)

        cb.save(fpath_out)
    else:
        plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0)

for subject, stim, method, project_dir, suffix, stc_type, src_to, cbar_type, clim \
    in zip(subjects, stims, methods, project_dirs, suffixes, stc_types, src_tos,
           cbars, clims):
    plot_result_estimates(subject, stim, method, project_dir, suffix, stc_type,
                          src_to, cbar_type, overwrite = overwrite, clim = clim)