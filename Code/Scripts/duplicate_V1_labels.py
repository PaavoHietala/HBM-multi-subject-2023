import shutil

exclude = [5, 8, 13, 15]
subjects = ['MEGCI_S' + str(idx) for idx in list(range(1,25)) if idx not in exclude]
subjects.append('fsaverage')

base = '/m/nbe/scratch/megci/data/FS_Subjects_MEGCI/'

for d in [base + s + '/label/' for s in subjects]:
    shutil.copyfile(d + 'lh_V1.label', d + 'lh.V1.label')
    shutil.copyfile(d + 'rh_V1.label', d + 'rh.V1.label')