# Improving source estimation of retinotopic MEG responses with joint analysis of multiple subjects.

This repository contains the analysis code for the manuscript "Improving source estimation of retinotopic MEG responses with joint analysis of multiple subjects." by Hietala et al. (year) submitted to \<journal>.

The analysis pipeline is adapted for a particular multifocal retinotopic mapping dataset which has been preprocessed with MATLAB, but the source estimation functions can be modified for other data with relative ease.

Currently supported MEG processing pipelines:
- eLORETA with Euclidean source-space averaging (Inspired by [Larson et al. 2014](https://doi.org/10.3389/fnins.2014.00330))
- Minimum Wasserstein estimates ([Janati et al. 2020](https://doi.org/10.1016/j.neuroimage.2020.116847))
- MWE with Euclidean source-space averaging

## Installation

1. Install the latest version of Python3 on your machine
2. Download/clone this repository on your machine with `git clone <repository url>.git` 
3. Install the required python packages listed below.

### Required packages

* Required Python packages are listed in `requirements.txt` and can be installed to your environment with e.g. pip:
    ```
    python -m pip install -r requirements.txt
    ```

* Additionally, a version of CuPy corresponding to the version of your CUDA driver is required to use CUDA acceleration with reMTW. With Linux you can check your CUDA driver version with the command `nvidia-smi`. The CUDA version is shown in the top right corner of the output. For more information visit [the cupy website](https://docs.cupy.dev/en/stable/install.html).

* The GroupMNE package, which includes the reMTW method, is not available in Python package repositories and has to be downloaded from [the GitHub page](https://hichamjanati.github.io/groupmne/). You can download the package using Git:
    ```bash
    git clone https://github.com/hichamjanati/groupmne
    cd groupmne
    python setup.py develop
    ```
    
    You can also "install" the package without permissions by dropping the root directory of the GroupMNE repository (`groupmne`) to the `Code` directory of this repository.

* In order to use the reMTW pipeline, you need to fix the bug in GroupMNE 0.0.1dev as per the instructions in [this GitHub issue](https://github.com/hichamjanati/groupmne/issues/24) (pull request pending). In a nutshell, the culprit is on line 124 of `groupmne\groupmne\group_model.py` preventing an elementwise addition to offset data on the 2nd hemisphere:
    ```python
    col_0 = i * n_sources[0]
    ```
    has to be changed to (typecast `n_sources` to numpy array instead of a list):
    ```python
    col_0 = i * np.array(n_sources[0])
    ```

## Usage

### Input data format

Three files are required per subject. The input file naming is not strict, as each file is explicitly identified in the pipeline settings.

1. Sensor-level recordings as `mne.Evoked` in MNE-Python's `.fif` file format. The input is expected to be preprocessed (including e.g. artefact suppression, averaging etc.) before these pipelines are run. An example of transforming a 3D MATLAB array (stimuli x time x sensors) to `mne.Evoked` can be found in `Code/Scripts/mat_to_mne_Evoked.py`.

2. Resting-state raw recording for noise covariance estimation.

3. MEG-MRI coregistration/transformation file.

### Running locally / on VDI

* For "classic" MNE analysis, open `Code/pipeline_classic.py` and set desired settings in the file. The settings are documented in the file. The MNE pipeline doesn't currently have any command line options. The pipeline can be run with\
`python Code/pipeline_classic.py`

* For reweighted Minimum Wasserstein Estimate (MWE0.5 or reMTW), open Code/pipeline_reMTW.py and set desired settings in the file. Some of the settings can also be used with command line options. Run with\
`python Code/pipeline_reMTW.py <command line options>`

* For reMTW with averaging run the `Code/pipeline_reMTW.py` with `average_stcs_source_space` toggle enabled.

### Running on a headless server or a HPC cluster

* If your machine has no displays the code has to be run with a virtual display e.g. xvfb.\
`xvfb-run python Code/pipeline_reMTW.py <command line options>`

* If used in a slurm-based HPC environment a GPU node has to be requested when creating the job with slurm option `--gres=gpu:1`.

* Array jobs with logging are the preferred way to run the code. An example array job file can be found in `Code/Scripts/MFinverse_slurm_array.sh`. If you are running reMTW as an array job, follow these steps (Snakemake or Nextflow would be optimal...):
    ```
    1. Run a single job with toggles from prepare_directories to compute_covariance_matrix enabled.
    2. Run an array of jobs with toggles estimate_source_timecourse, morph_to_fsaverage and average_stcs_source_space enabled.
    3. Run a single job for tabluate_geodesics.
    ```

### reMTW command line options

The following command line options can be used with `pipeline_reMTW.py`:

`-stim=<stim name>, ...` solve for specific stimuli (1-n), e.g. sector9\
`-alpha=<number>` set fixed alpha and skip alpha search\
`-beta=<number>` set fixed beta and skip beta search\
`-hyperplot` create alpha and beta plots with equally distributed points (Figure 2)\
`-time=<start>,<stop>` crop evoked responses to this timeframe. If stop is omitted, stop=start.\
`-target=<num>` Number of active source points to aim for.\
`-suffix=<str>` Suffix to append to all file names, e.g. 20subjects.\
`-concomitant=<bool>` Use concomitant noise level estimation True/False.\
`-dir=<path>` Use different project directory than defined in pipeline.

## Directory tree

This repository is built with the following directory tree:

```
.
├── Code/
│   ├── Core                    # Pipeline functions
│   ├── Scripts                 # Plotting, data transformation etc.
│   ├── groupmne                # GroupMNE package, see Installation
│   ├── pipeline_classic.py     # eLORETA + avg pipeline
│   └── pipeline_reMTW.py       # MWE and MWE + avg pipeline
├── project_dir/                # pipeline-specific project folder
│   └── Data/
│       ├── avg                 # Source-space-averaged source estimates
│       ├── cov                 # Subjects' noise covariance matrices
│       ├── fwd                 # Subjects' forward solutions
│       ├── inv                 # Subjects' inverse solutions
│       ├── plot                # Figures, parameter logs etc.
│       ├── slurm_out           # Slurm / HPC output and error logs
│       ├── src                 # Subjects' source spaces
│       ├── stc                 # Source estimates for individual subjects
│       └── stc_m               # Subjects' source estimates morphed to fsaverage
├── README.md
└── requirements.txt
```

Both pipelines have their own project_dir and the name/path is set in pipeline settings.

## Potential issues

* If you encounter the following error on Linux-based HPC (quite rare and random, but annoying)\
`RuntimeError: cannot cache function 'bincount': no locator available for file...`\
you can circumvent it by explicitly setting the Numba cache:\
`export NUMBA_CACHE_DIR=/tmp/`