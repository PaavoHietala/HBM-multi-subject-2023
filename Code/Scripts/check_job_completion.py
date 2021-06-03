'''
Check that all the jobs in the array have finished correctly by reading
the log files and looking for required lines in order.
'''

folder = '/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/'
files = ['15subj_60895489_' + str(i) + '.out' for i in range(0,24)]

search = ['Got aMax',
          'Got beta_=',
          'morphing to fsaverage',
          'Averaging stcs in source space',
          '[done]']

for f in files:
    search_temp = search.copy()

    with open(folder + f, 'r') as f_in:
        lines = f_in.readlines()
    
    for line in lines:
        if line.startswith(search_temp[0]):
            search_temp.pop(0)
        if len(search_temp) == 0:
            break

    if len(search_temp) != 0:
        print('Following lines were not found in ' + f + '\n' + '\n'.join(search_temp))
