'''
Plot freesurfer's V1 label and manual V1 label in a similar fashion to all 
other brain plots in the Thesis. The result is in appendix B
'''

import mne
import matplotlib.pyplot as plt
import numpy as np

subjects_dir = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'
mne.set_config('SUBJECTS_DIR', subjects_dir)

out_dir = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/'

# Create a dummy stc to access more refined brain plotting tools
stc = mne.SourceEstimate(np.array([[0], [0]]),
                        [np.array([0]), np.array([0])], 0, 0)

for label in ['.V1.label', '.V1_exvivo.label']:
    brain = stc.plot(subject = 'MEGCI_S1', hemi = 'split', surface = 'inflated',
                     background = 'w', size = (1500, 600), colorbar = False,
                     time_viewer = False, show_traces = False)

    for hemi in ['lh', 'rh']:
        v1 = mne.read_label(subjects_dir + 'MEGCI_S1/label/' + hemi + label,
                            'MEGCI_S1')
        brain.add_label(v1, borders = 3)

    brain.show_view({'elevation' : 100, 'azimuth' : -60}, distance = 400, col = 0)
    brain.show_view({'elevation' : 100, 'azimuth' : -120}, distance = 400, col = 1)

    screenshot = brain.screenshot()
    brain.close()

    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    fig, axes = plt.subplots(num = label, figsize = (8, 8.))

    axes.imshow(cropped_screenshot)
    axes.axis('off')
    fig.subplots_adjust(left = 0.15, right = 0.9, bottom = 0.01, top = 0.9,
                        wspace = 0.1, hspace = 0.5)

    print('Saving to ', out_dir + label + '.png')
    plt.savefig(out_dir + label[1:] + '.png', bbox_inches = 'tight', pad_inches = 0.0)