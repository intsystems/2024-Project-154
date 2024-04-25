# import seaborn as sns
import glob
import json
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn import base

# generate plots from all the different results and save them in the figures folder

# load the results
base_results_folder = os.path.abspath("")
os.makedirs(os.path.join(base_results_folder, 'figures'), exist_ok=True)
plot_dilation = True


if plot_dilation:
    # dilation model, match mismatch results
    # plot boxplot of the results per window length
    # load evaluation results for all window lengths

    files = glob.glob(os.path.join(base_results_folder, "baseline_tensorflow/results_dilated_convolutional_model_4_MM_5_s_envelope/eval_*.json"))
    print(os.path.join(base_results_folder, "results_dilated_convolutional_model_4_MM_5_s_envelope/eval_*.json"))
    # sort the files
    files.sort()

    # create dict to save all results per sub
    results = []
    windows = []
    number_mismatch = []
    for f in files:

        # load the results
        with open(f, "rb") as ff:
            res = json.load(ff)
        #loop over res and get accuracy in a list
        acc = []

        for sub, sub_res in res.items():
            if 'compile_metrics' in sub_res:
                acc.append(sub_res['compile_metrics'])

        results.append(acc)

        # get the window length
        windows.append(int(f.split("_")[-2].split(".")[0]))
        number_mismatch.append(int(f.split("_")[-3].split(".")[0]))

    # sort windows and results according to windows
    windows, results, number_mismatch = zip(*sorted(zip(windows, results, number_mismatch)))

    #boxplot of the results
    plt.boxplot(results, labels=[*zip(windows, number_mismatch)])
    plt.xlabel("(Window length, Number of mismatch)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of dilation model, per window length")
    plt.savefig(os.path.join(base_results_folder, 'baseline_tensorflow/figures', "boxplot_dilated_conv.pdf"))