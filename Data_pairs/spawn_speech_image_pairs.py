#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
#
# This script spawns all speech and image positive pair generation for all specified speech and image 
# datasets and subsets. 
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import os
import datetime
from os import path
import math
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import re
import itertools
import subprocess

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")
from paths import data_path
from paths import pair_path
from paths import feats_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path
data_path = path.join("..", data_path)
pair_path = path.join("..", pair_path)
feats_path = path.join("..", feats_path)

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import model_legos_library

PAIR_DATASETS = [("TIDigits", "MNIST")]
DATASET_TYPES = ["train", "validation", "test"]
METRIC = ["cosine"]

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    feat_fns = []

    for dataset_type in DATASET_TYPES:
        for (speech_dataset, image_dataset) in PAIR_DATASETS:
            speech_dataset_type = "val" if dataset_type == "validation" else dataset_type
            speech_fn = path.join(feats_path, speech_dataset, "Subsets", "Words", "mfcc", "gt_" + speech_dataset_type + "_mfcc.npz")
            image_fn = path.join(feats_path, image_dataset, dataset_type + ".npz")
            feat_fns.append((speech_fn, image_fn))

    for ((sp_fn, im_fn), metr) in list(itertools.product(feat_fns, METRIC)):  

        cmd = "./classifier_latents.py " + " --feats_fn {}".format(sp_fn)
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

        cmd = "./classifier_latents.py " + " --feats_fn {}".format(im_fn)
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        
        cmd = "./speech_image_pairs.py " + " --speech_feats_fn {} --image_feats_fn {} --metric {}".format(sp_fn, im_fn, metr)
        print_string = cmd
        model_setup_library.command_printing(print_string)
        sys.stdout.flush()
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()

if __name__ == "__main__":
    main()