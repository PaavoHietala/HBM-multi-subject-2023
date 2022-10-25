'''
Create a histogram of active source point counts from all log files in /reMTW/Data/plot/
'''

import os
import matplotlib.pyplot as plt

basedir = "/m/nbe/scratch/megci/MFinverse/reMTW-data-thesis-submission-state/Data/plot"

def find_logs(dirpath):
    '''
    Return a list of .txt files in given dirpath.

    Parameters
    ----------
    dirpath : str
        Directory for the text file search

    Returns
    -------
    list of str
        Complete paths of all .txt files in given dirpath
    '''

    return [os.path.join(basedir, file)
            for file in os.listdir(dirpath)
            if file.endswith(".txt")]

def find_numbers(fpath):
    '''
    Extract the numbers of active source points in each log file.

    Parameters
    ----------
    fpath : str
        Complete path to the file being analyzed.

    Returns
    -------
    list of float
        List of all source point numbers from the log file.
    '''

    numbers = []

    with open(fpath) as f:
        log = f.readlines()
    
    for idx in range(len(log)):
        if log[idx].startswith("Active source points") and log[idx + 1].strip():
            numbers += [float(number) for number in log[idx + 1].split(", ")]
    
    return numbers


if __name__ == "__main__":
    files = find_logs(basedir)
    
    numbers = []
    for file in files:
        numbers += find_numbers(file)
    
    plt.figure()
    plt.hist(numbers, bins = [(x / 2) - 0.25 for x in range(0, 21)], rwidth = 0.9)
    plt.xticks([x for x in range(0, 11)])
    plt.savefig(os.path.join(basedir, "sourcepoint_number_histogram.png"))
            