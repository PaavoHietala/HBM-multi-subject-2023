# MFInverse

Improving MEG retinotopical mapping accuracy with multi-subject joint analysis.

Adapted for particular multifocal retinotopic mapping dataset which has been preprocessed with MATLAB, but the functions can be modified for other data with relative ease.

## Usage

For "classic" MNE analysis, open `Code/pipeline_classic.py` and set desired settings in the file.
Run with `python Code/pipeline_classic.py`.

For reweighted Minimum Wasserstein Estimate (MWE0.5 or reMTW), open `Code/pipeline_reMTW.py` and set desired settings in the file. Run with `python Code/pipeline_reMTW.py`.

### reMTW command line options

`-stim=<stim name>,<stim name>` solve for specific stimuli (1-n), e.g. sector9\
`-stimnum=<stum number>,<stim number>` same as -stim, but input only the number, e.g. 9\
`-alpha=<number>` set fixed alpha and skip alpha search\
`-beta=<number>` set fixed beta and skip beta search\
`-tenplot` create alpha and beta plots with 11 equally distributed points\
`-time=<start>,<stop>` crop evoked responses to this timeframe

### Running locally

### Running on triton

### Required packages