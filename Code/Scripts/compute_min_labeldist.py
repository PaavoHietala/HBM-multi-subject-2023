import sys
import os.path as op
import mne
import numpy as np

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core.utils import get_fname

# A script for computing the minimum achievable distance between an active
# dipole and V1 label vertices, corresponding to the medoid (center of minimum
# distance).

subject = 'fsaverage'
src_spacing = 'ico4'
subjects_dir = mne.get_config('SUBJECTS_DIR')
project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'

#-------------------------------------------------------------------------------

fname_src = get_fname(subject, 'src', src_spacing = src_spacing)
src = mne.read_source_spaces(op.join(project_dir, 'Data', 'src', fname_src),
                             verbose = False)

distances = []
for idx, hemi in enumerate(['lh', 'rh']):
    label = mne.read_label(op.join(subjects_dir, subject, 'label', 
                                   f'{hemi}.V1_exvivo.label'), subject)
    used_verts = label.get_vertices_used(src[idx]['vertno'])
    distances.append(np.zeros((len(used_verts), len(used_verts))))

    for row_i, row_vert in enumerate(used_verts):
        for col_i, col_vert in enumerate(used_verts):
            distances[idx][col_i, row_i] = src[idx]['dist'][col_vert, row_vert]
    
    distances[idx] = distances[idx] * 1000
    
    print(f'Lowest mean on {hemi}: {np.min(np.mean(distances[idx], axis = 0)):.1f}')
    print(f'Lowest median on {hemi}: {np.min(np.median(distances[idx], axis = 0)):.1f}')
    print(f'Lowest STD on {hemi}: {np.min(np.std(distances[idx], axis = 0)):.1f}\n')

    distances[idx] = distances[idx][np.triu_indices_from(distances[idx])]
    
    print(f'Mean on {hemi}: {np.mean(distances[idx]):.1f}')
    print(f'Median on {hemi}: {np.median(distances[idx]):.1f}')
    print(f'STD on {hemi}: {np.std(distances[idx]):.1f}\n')

agg = np.concatenate((distances[0][:], distances[1][:]))
print(f'Grand mean: {np.mean(agg):.1f}, median: {np.median(agg):.1f},'
      + f'std: {np.std(agg):.1f}')
