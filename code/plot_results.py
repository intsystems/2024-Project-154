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

# generate plots from all the different results and plsave them in the figures folder

# load the results
# base_results_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code')
base_results_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base_results_folder, 'figures'), exist_ok=True)
plot_dilation = True
plot_linear_backward = False
plot_linear_forward = False
plot_vlaai = False

freq_bands = {
    'Delta [0.5-4]': (0.5, 4.0),
    'Theta [4-8]': (4, 8.0),
    'Alpha [8-14]': (8, 14.0),
    'Beta [14-30]': (14, 30.0),
    'Broadband [0.5-32]': (0.5, 32.0),
}

if plot_dilation:
    # dilation model, match mismatch results
    # plot boxplot of the results per window length
    # load evaluation results for all window lengths

    files = glob.glob(os.path.join(base_results_folder, "code/results_dilated_convolutional_model_5_MM_5_s_envelope/eval_*.json"))
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
        # print(f.split("_")[-2].split(".")[0])
        windows.append(int(f.split("_")[-2].split(".")[0]))
        number_mismatch.append(int(f.split("_")[-3].split(".")[0]))

    # sort windows and results according to windows
    # print(results)
    windows, results, number_mismatch = zip(*sorted(zip(windows, results, number_mismatch)))

    #boxplot of the results
    plt.boxplot(results, labels=[*zip(windows, number_mismatch)])
    plt.xlabel("(Window length, Number of mismatch)")
    plt.ylabel("F1-score (%)")
    # plt.title("Accuracy of dilation model, per window length")
    plt.savefig(os.path.join(base_results_folder, 'figures', "boxplot_dilated_conv.pdf"))

