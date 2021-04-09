# MFInverse

Improving MEG retinotopical mapping accuracy with multi-subject joint analysis.

Adapted for particular multifocal retinotopic mapping dataset which has been preprocessed with MATLAB, but the functions can be modified for other data with relative ease.

Currently supported MEG processing pipelines:
- MNE estimate with Euclidean average (Larson et al. 2014; https://doi.org/10.3389/fnins.2014.00330)
- Minimum Wasserstein estimates (Janati et al. 2020; https://doi.org/10.1016/j.neuroimage.2020.116847)

## Installation

Download/clone this repository on your machine and install the packages listed below.

Please note the critical bugfixes in MNE and GroupMNE that have not yet been implemented in the package distributions.

### Required packages

Required Python packages are listed in `requirements.txt`. Additionally, a version of CuPy corresponding to the version of your CUDA driver is required to use CUDA acceleration with reMTW. For example, `cupy-cuda112` is used with Triton GPU nodes and `cupy-cuda100` with Ubuntu nVidia VDI. Check your CUDA driver version with the command `nvidia-smi`.

The GroupMNE package, which includes the reMTW method, is not available in Python package repositories and has to be downloaded from\
https://hichamjanati.github.io/groupmne/

In order to use the reMTW pipeline, you need to fix the bug in GroupMNE 0.0.1dev as per the instructions in this GitHub issue (pull request pending)\
https://github.com/hichamjanati/groupmne/issues/24

In order to use the function `expand_peak_labels()` in `Core/visualize.py` you need to fix a bug in MNE 0.22.0 by modifying the function `grow_labels()` in file `mne/label.py` as per the GitHub issue at\
https://github.com/mne-tools/mne-python/issues/8848

You might have to load the relevant GPU libraries on Triton with\
`ml anaconda3/2019.11-gpu` and\
`ml cuda/10`\

On VDI\
`ml anaconda3/2019.08.19-gpu`

## Usage

### Running locally / on VDI

For "classic" MNE analysis, open Code/pipeline_classic.py and set desired settings in the file. The MNE pipeline doesn't currently have any command line options. The pipeline can be run with\
`python Code/pipeline_classic.py`

For reweighted Minimum Wasserstein Estimate (MWE0.5 or reMTW), open Code/pipeline_reMTW.py and set desired settings in the file or with command line options. Run with\
`python Code/pipeline_reMTW.py <command line options>`

### Running on triton

Triton has no displays, so the code has to be run with a virtual display e.g. xvfb.\
`xvfb-run python Code/pipeline_reMTW.py <command line options>`

Additionally, when creating the job a GPU node has to be requested with slurm option `--gres=gpu:1`.

An example array job file can be found in Code/Scripts/MFinverse_slurm_array.sh.

### reMTW command line options

`-stim=<stim name>,<stim name>` solve for specific stimuli (1-n), e.g. sector9\
`-stimnum=<stum number>,<stim number>` same as -stim, but input only the number, e.g. 9\
`-alpha=<number>` set fixed alpha and skip alpha search\
`-beta=<number>` set fixed beta and skip beta search\
`-tenplot` create alpha and beta plots with 11 equally distributed points\
`-time=<start>,<stop>` crop evoked responses to this timeframe. If stop is omitted, stop=start.

## Potential issues

If you encounter the following error on Triton (quite rare and random, but annoying)\
`RuntimeError: cannot cache function 'bincount': no locator available for file...`\
you can circumvent it by explicitly setting the Numba cache:\
`export NUMBA_CACHE_DIR=/tmp/`