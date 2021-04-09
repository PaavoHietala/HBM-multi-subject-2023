'''
Get aMax values from slurm out files and print them divided by 2 to get safe
alphas for all stimuli.
'''

job_id = '60025086_'
files = ['/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/' + job_id + str(i) + '.out' for i in range(0,24)]
alphas = []

for fname in files:
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Got aMax='):
                alphas.append(float(line.rstrip()[9:]) / 2)

print(alphas)
