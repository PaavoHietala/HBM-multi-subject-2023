'''
This script plots the 8x3 source estimate grids in Figures 5-7. The script
has to be run separately for each figure, changing the script settings in
between.

The output image grid is saved to
<project_dir>/Data/plot/EstimateGrid-<method>_<stc_type>_<suffix>.png
And the grid with the colorbar attached to
<project_dir>/Data/plot/EstimateGrid-<method>_<stc_type>_<suffix>-cb.png
'''

import os.path as op
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core import utils

from mpl_toolkits.axes_grid1 import (make_axes_locatable)

# Script settings --------------------------------------------------------------

# Root data directory of the project, str

project_dir = '/m/nbe/scratch/megci/MFinverse/Classic/Data/'

# Subjects' MRI location (FreeSurfer's subject dir), str

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

stc_type = 'avg'

# Inverse solution method, 'remtw' or 'eLORETA'

method = 'eLORETA'

# V1 freesurfer label location

label_fpath_lh = subjects_dir + 'fsaverage/label/lh.V1_exvivo.label'
label_fpath_rh = subjects_dir + 'fsaverage/label/rh.V1_exvivo.label'

# Run the script ---------------------------------------------------------------

fpath_out = op.join(project_dir, 'plot', f'EstimateGrid-{method}-{stc_type}-{suffix}.png')

stcs = []
abs_max = 0
abs_min_max = 1e10
abs_min = 1e100

for stim in stims:
    # Load the source estimate to the list of stcs
    if stc_type == 'avg':
        fname_stc = '-'.join([subject, src_spacing, method, 'f', stim, suffix])
    else:
        fname_stc = '-'.join([og_subject, src_spacing, method, subject, 'f',
                              stim, suffix])
    fpath_stc = op.join(project_dir, stc_type, fname_stc)
    stc = mne.read_source_estimate(fpath_stc, subject = subject)
    stcs.append(stc)

    # Get maximum and minimum amplitudes for thresholding/colorbars
    if np.max(abs(stc.data)) > abs_max:
        abs_max = np.max(abs(stc.data))
    if np.max(abs(stc.data)) < abs_min_max:
        abs_min_max = np.max(abs(stc.data))
    if np.min(abs(stc.data)[np.nonzero(stc.data)]) < abs_min:
        abs_min = np.min(abs(stc.data)[np.nonzero(stc.data)])

fig, axes = plt.subplots(nrows = 8, ncols = 3, figsize = (8, 24))

# Set colorbar properties depending on the data type (averaged or not)
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
        
        peaks, peak_hemis = utils.find_peaks(project_dir, src_spacing, method,
                                             'f', [stims[stim_idx]], bilaterals,
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
        
        for label in [label_fpath_lh, label_fpath_rh]:
            v1 = mne.read_label(label, 'fsaverage')
            brain.add_label(v1, borders = 2, color = 'lime')
        
        # Add peak foci
        for peak, hemi in zip(peaks, peak_hemis):
            brain.add_foci(peak, coords_as_verts = True, scale_factor = 1,
                           color = 'lime', hemi = hemi)
        
        brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 400, col = 0)
        brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 400, col = 1)
        screenshot = brain.screenshot()
        brain.close()
        cropped_screenshot = utils.crop_whitespace(screenshot)

        axes[row_idx][col_idx].imshow(cropped_screenshot)
        axes[row_idx][col_idx].axis('off')

plt.tight_layout()     
plt.savefig(fpath_out, bbox_inches = 'tight', pad_inches = 0.0, dpi = 300)

# Create a separate colorbar ---------------------------------------------------

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

cbar.outline.set_visible(True)
plt.tight_layout()

# Save colorbar to buffer
buf = BytesIO()
fig.savefig(buf, dpi = 300)
buf.seek(0)
cb = Image.open(buf)

# Crop whitespace around the colorbar
screenshot = np.asarray(cb)
cropped_screenshot = utils.crop_whitespace(screenshot, borders_only = True)
cb = Image.fromarray(cropped_screenshot)

# Load the source estimate grid and add colorbar to the bottom of the image
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
