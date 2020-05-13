#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script generates speech-image positive pairs from latents extracted from a Siamese or 
# classifier model.
#

from __future__ import division
from __future__ import print_function
import argparse
import sys
from tqdm import tqdm
import numpy as np
import os
import datetime
from os import path
import math
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import re
import pickle
import tensorflow as tf
import subprocess

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("..")
from paths import data_path
from paths import pair_path
from paths import data_lib_path
from paths import general_lib_path
from paths import model_lib_path
from paths import few_shot_lib_path
data_path = path.join("..", data_path)
pair_path = path.join("..", pair_path)

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", general_lib_path))
import util_library

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import model_legos_library


sys.path.append(path.join("..", few_shot_lib_path))
import few_shot_learning_library

SPEECH_DATASETS = ["TIDigits"]
IMAGE_DATASETS = ["MNIST"]
COL_LENGTH = model_setup_library.COL_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Argument function
#
#_____________________________________________________________________________________________________________________________________


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech_feats_fn", type=str)
    parser.add_argument("--image_feats_fn", type=str)
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "euclidean_squared"], default="cosine")
    return parser.parse_args()

#_____________________________________________________________________________________________________________________________________
#
# Main
#
#_____________________________________________________________________________________________________________________________________


def main():

    args = check_argv()

    if args.metric == "cosine":
        dist_func = "cosine"
    elif args.metric == "euclidean":
        dist_func = "euclidean"
    elif args.metric == "euclidean_squared":
        dist_func == "sqeuclidean"

    print("Start time: {}".format(datetime.datetime.now()))
    if args.speech_feats_fn.split("_")[-2] == args.image_feats_fn.split("/")[-1].split('.')[0]:
        subset = args.image_feats_fn.split("/")[-1].split('.')[0]
    elif args.speech_feats_fn.split("_")[-2] == "val" and args.image_feats_fn.split("/")[-1].split('.')[0] == "validation":
        subset = args.image_feats_fn.split("/")[-1].split('.')[0]
    else: sys.exit(0)

    speech_not_image_pairs = True if args.speech_feats_fn.split("/")[2] in SPEECH_DATASETS and args.image_feats_fn.split("/")[2] in IMAGE_DATASETS else "INVALID" 
    if speech_not_image_pairs == "INVALID":
        print("Specified dataset to get pairs for, not valid.")
        sys.exit(0)    

    key_pair_file = path.join(pair_path, args.speech_feats_fn.split("/")[2] + "_speech_" + args.image_feats_fn.split("/")[2] + "_image_pairs")
    model = "classifier"
    util_library.check_dir(key_pair_file)

    if os.path.isfile(key_pair_file) is False:

        speech_latent_npz = path.join(pair_path, "/".join(args.speech_feats_fn.split(".")[-2].split("/")[2:]), model + "_latents", model + "_feats.npz")
        image_latent_npz = path.join(pair_path, "/".join(args.image_feats_fn.split(".")[-2].split("/")[2:]), model + "_latents", model + "_feats.npz")


        image_latents, image_keys = data_library.load_latent_data_from_npz(image_latent_npz)
        image_latents = np.asarray(image_latents)
        image_latents = np.squeeze(image_latents)

        speech_latents, speech_keys = data_library.load_latent_data_from_npz(speech_latent_npz)
        speech_latents = np.asarray(speech_latents)
        speech_latents = np.squeeze(speech_latents)


        speech_latents = (speech_latents - speech_latents.mean(axis=0))/speech_latents.std(axis=0)
        image_latents = (image_latents - image_latents.mean(axis=0))/image_latents.std(axis=0)
        

        im_x, im_labels, im_keys = (
            data_library.load_image_data_from_npz(args.image_feats_fn)
            )

        sp_x, sp_labels, sp_lengths, sp_keys = (
            data_library.load_speech_data_from_npz(args.speech_feats_fn)
            )
        max_frames = 100
        d_frame = 13
        print("\nLimiting dimensionality: {}".format(d_frame))
        print("Limiting number of frames: {}\n".format(max_frames))
        data_library.truncate_data_dim(sp_x, sp_lengths, d_frame, max_frames)



        support_set = few_shot_learning_library.construct_few_shot_support_set_with_keys(
                sp_x, sp_labels, sp_keys, sp_lengths, im_x, im_labels, im_keys, 11, 5
                )

        support_set_speech_keys = [] 
        support_set_image_keys = [] 
        support_set_speech_latents = []
        support_set_image_latents = []

        for key in support_set: 
            support_set_speech_keys.extend(support_set[key]["speech_keys"])
            for sp_key in support_set[key]["speech_keys"]:
                ind = np.where(np.asarray(speech_keys) == sp_key)[0][0]
                support_set_speech_latents.append(speech_latents[ind, :])

            support_set_image_keys.extend(support_set[key]["image_keys"])
            for im_key in support_set[key]["image_keys"]:
                ind = np.where(np.asarray(image_keys) == im_key)[0][0]
                support_set_image_latents.append(image_latents[ind, :])

        support_set_speech_latents = np.asarray(support_set_speech_latents)
        support_set_image_latents = np.asarray(support_set_image_latents)

        support_dict = {}
        already_used = []

        for key in support_set_speech_keys:
            label = key.split("_")[0]

            for im_key in support_set_image_keys:
                if few_shot_learning_library.label_test(im_key.split("_")[0],  label) and im_key not in already_used:
                    support_dict[key] = im_key
                    already_used.append(im_key)
                    break


        speech_dict = {}
        speech_distances = cdist(speech_latents, support_set_speech_latents, dist_func)
        speech_indexes = np.argsort(speech_distances, axis=1)
        for i, sp_key in enumerate(speech_keys):
            if sp_key not in support_set_speech_keys:
                for count in range(speech_indexes.shape[-1]):
                    ind = speech_indexes[i, count]
                    speech_dict[sp_key] = support_dict[support_set_speech_keys[ind]]
                    break

        image_dict = {}
        image_distances = cdist(image_latents, support_set_image_latents, dist_func)
        image_indexes = np.argsort(image_distances, axis=1)

        for i, im_key in enumerate(image_keys):

            if im_key not in support_set_image_keys:

                for count in range(image_indexes.shape[-1]):
                    ind = image_indexes[i, count]

                    if support_set_image_keys[ind] not in image_dict: 
                        image_dict[support_set_image_keys[ind]] = []

                    image_dict[support_set_image_keys[ind]].append(im_key)
                    break

        already_used_im_keys = []
        key_pair_file = open(path.join(key_pair_file, subset + "_pairs.txt"), 'w')
        for sp_key in tqdm(speech_dict, desc="Generating speech-image pairs", ncols=COL_LENGTH):
            possible_im_keys = image_dict[speech_dict[sp_key]]

            for i in range(len(possible_im_keys)):
                possible_key = possible_im_keys[i]
                if possible_key not in already_used_im_keys:
                    key_pair_file.write(f'{sp_key}\t{possible_key}\n')
                    already_used_im_keys.append(possible_key)
                    image_dict[speech_dict[sp_key]].remove(possible_key)
                    break

        key_pair_file.close()
    print("End time: {}".format(datetime.datetime.now()))
if __name__ == "__main__":
    main()