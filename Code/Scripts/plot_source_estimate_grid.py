'''
This script plots the 8x3 source estimate grids in Appendix B.
'''

import os.path as op
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from io import BytesIO

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core import utils

from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)

# Root data directory of the project, str

project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/Data/'

# Subjects' MRI location, str

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

# Subject of the stcs to draw

subject = 'fsaverage'

# Original subject if stc_m is plotted

og_subject = 'MEGCI_S1'

# List of stimuli and bilateral stimuli

stims = ['sector' + str(i) for i in range(1,25)]
bilaterals = ['sector' + str(i) for i in [3, 7, 11, 15, 19, 23]]

# Suffix used in stcs, usually subject count

suffix = '20subjects'

# Source spacing used in stcs, e.g. 'ico4'

src_spacing = 'ico4'

# Stc type, either 'stc', 'stc_m' or 'avg'

stc_type = 'stc_m'

# Inverse solution method, e.g. 'remtw'

method = 'remtw'

#
# Run the script ---------------------------------------------------------------
#

fpath_out = op.join(project_dir, 'plot', '-'.join(['EstimateGrid', method, stc_type, suffix]) + '.png')

stcs = []

abs_max = 0
abs_min_max = 1e10
abs_min = 1e100
for stim in stims:
    if stc_type == 'avg':
        fname_stc = '-'.join([subject, src_spacing, method, 'f', stim, suffix])
    else:
        fname_stc = '-'.join([og_subject, src_spacing, method, subject, 'f', stim, suffix])
    fpath_stc = op.join(project_dir, stc_type, fname_stc)
    stc = mne.read_source_estimate(fpath_stc, subject = subject)
    stcs.append(stc)
    if np.max(abs(stc.data)) > abs_max:
        abs_max = np.max(abs(stc.data))
    if np.max(abs(stc.data)) < abs_min_max:
        abs_min_max = np.max(abs(stc.data))
    if np.min(abs(stc.data)[np.nonzero(stc.data)]) < abs_min:
        abs_min = np.min(abs(stc.data)[np.nonzero(stc.data)])

fig, axes = plt.subplots(nrows = 8, ncols = 3, figsize = (8,24))
                         #gridspec_kw = {'height_ratios' : [1] * 8 + [0.00001]})

if method == 'remtw' and stc_type == 'stc_m':
    clim = {'kind' : 'value', 'pos_lims' : [0, 0, abs_min_max]}
elif method == 'remtw' and stc_type == 'avg':
    clim = {'kind' : 'value', 'lims' : [0, 0, 1.5 * abs_min_max]}
else:
    clim = {'kind' : 'value', 'lims' : [0. * abs_max, 0.3 * abs_max, 0.8 * abs_max]}

for row_idx in range(0,8):
    for col_idx in range(0,3):
        stim_idx = row_idx + col_idx * 8

        stc = stcs[stim_idx]
        
        peaks, peak_hemis = utils.find_peaks(project_dir, src_spacing, method, 'f',
                                             [stims[stim_idx]], bilaterals,
                                             suffix, stc = stc)

        # Set one point to very low value in remtw to avoid
        # all-blue plot caused by all-zero estimate
        if method == 'remtw':
            if np.count_nonzero(stc.lh_data) == 0:
                stc.lh_data[0] = 1e-20
            if np.count_nonzero(stc.rh_data) == 0:
                stc.rh_data[0] = 1e-20

        # Plot the STC with V1 borders, get the brain image, crop it:
        brain = stc.plot(hemi='split', size=(1500,600),
                         subject = subject,
                         initial_time = (0.08 if stc.data.shape[1] > 1 else None),
                         background = 'w', colorbar = False,
                         time_viewer = False, show_traces = False, clim = clim)
        
        for hemi in ['lh', 'rh']:
            v1 = mne.read_label('/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
                                + 'fsaverage/label/' + hemi + '.V1_exvivo.label',
                                'fsaverage')
            brain.add_label(v1, borders = 2, color = 'lime')
        
        # Add peak foci
        for peak, hemi in zip(peaks, peak_hemis):
            brain.add_foci(peak, coords_as_verts = True, scale_factor = 1,
                           color = 'lime', hemi = hemi)
        
        brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 400, col = 0)
        brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 400, col = 1)
        screenshot = brain.screenshot()
        brain.close()

        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

        axes[row_idx][col_idx].imshow(cropped_screenshot)
        axes[row_idx][col_idx].axis('off')

# Save image
plt.tight_layout()     
plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0)

#
# Create colorbar --------------------------------------------------------------
#

fig, ax = plt.subplots(figsize = (3, 4))
ax.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size = '5%', pad = 0.2)
cbar = mne.viz.plot_brain_colorbar(cax, clim = clim, orientation = 'horizontal',
                                   label = 'Activation (Am)')
cbar.ax.tick_params(labelsize = 12)
if method == 'remtw' and stc_type == 'stc_m':
    cbar.set_ticks([-clim['pos_lims'][2], 0, clim['pos_lims'][2]])
    cbar.set_ticklabels(["%.2g" % -clim['pos_lims'][2],
                         "%.2g" % clim['pos_lims'][0],
                         "%.2g" % clim['pos_lims'][2]])
elif method == 'remtw' and stc_type == 'avg':
    cbar.set_ticks([clim['lims'][0], (clim['lims'][2] / 2), clim['lims'][2]])
    cbar.set_ticklabels(["%.2g" % clim['lims'][0], "%.2g" % (clim['lims'][2] / 2),
                     "%.2g" % clim['lims'][2]])
else: 
    cbar.set_ticklabels(["%.2g" % clim['lims'][0], "%.2g" % clim['lims'][1],
                         "%.2g" % clim['lims'][2]])
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

# Load grid and join with cb
grid = Image.open(fpath_out)

w, h = zip(*(i.size for i in [grid, cb]))

final = Image.new('RGB', (w[0], sum(h) + 20), color = (255, 255, 255))

y_offset = 0
x_offset = 0
for im in [grid, cb]:
    final.paste(im, (x_offset, y_offset))
    y_offset += im.size[1] + 20
    x_offset += int((w[0] / 2) - (w[1] / 2))

final.save(fpath_out[:-4] + '-cb.png')






