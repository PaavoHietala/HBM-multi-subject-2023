'''
Get safe alpha and beta values from slurm out files and print them for all stimuli.
'''

slurm_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/5-subject/Data/slurm_out/'
job_id = '60361386'

files = [slurm_dir + job_id + '_' + str(i) + '.out' for i in range(0,24)]
alphas = []
betas = []

for fname in files:
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Got aMax='):
                alphas.append(float(line.rstrip()[9:]) / 2)
            elif line.startswith('Got beta_='):
                betas.append(float(line.rstrip()[10:]))

print(alphas)
print(betas)
