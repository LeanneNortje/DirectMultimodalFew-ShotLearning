#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains functions to setup or retrieve a speech-image model library from which the 
# model can be built, trained and tested. 
#

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import logging
import tensorflow as tf
import hashlib
import timeit
import sys
import subprocess
import pickle
from tqdm import tqdm
import re
import shutil
import model_setup_library

sys.path.append("..")
from paths import model_path
from paths import non_final_model_path
from paths import feats_path
from paths import data_path
from paths import general_lib_path
from paths import few_shot_lib_path
from paths import episodes
from paths import pair_path

model_path = path.join("..", model_path)
non_final_model_path = path.join("..", non_final_model_path)
feats_path = path.join("..", feats_path)
data_path = path.join("..", data_path)
few_shot_lib_path = path.join("..", few_shot_lib_path)
pair_path = path.join("..", pair_path)

sys.path.append(path.join("..", general_lib_path))
import util_library
PRINT_LENGTH =  180
COL_LENGTH =  PRINT_LENGTH - len("\t".expandtabs())
#_____________________________________________________________________________________________________________________________________
#
# Dataset variables
#
#_____________________________________________________________________________________________________________________________________

SPEECH_DATASETS = ["TIDigits", "buckeye"]
IMAGE_DATASETS = ["MNIST", "omniglot"]
DIGIT_LIST = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh"]
IMAGE_PAIR_TYPES = ["kmeans", "siamese", "classifier"]
SPEECH_PAIR_TYPES = ["siamese", "classifier"]

#_____________________________________________________________________________________________________________________________________
#
# Library setup
#
#_____________________________________________________________________________________________________________________________________

def base_library_setup(model_lib):
    
    model_lib["speech_train_fn"] = path.join(
        feats_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"],
        model_lib["train_tag"] + "_train_" + model_lib["features_type"] + ".npz"
        )

    model_lib["speech_pair_dir"] = path.join(
        pair_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"],
        model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
        )
    model_lib["speech_neg_pair_dir"] = path.join(
        pair_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"],
        model_lib["train_tag"] + "_train_" + model_lib["features_type"], "key_" + model_lib["pair_type"] + "_negatives.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_negatives.list"
        )

    model_lib["speech_val_fn"] = path.join(
        feats_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"], 
        model_lib["train_tag"] + "_val_" + model_lib["features_type"] + ".npz"
        )

    model_lib["val_speech_pair_dir"] = path.join(
        pair_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"], 
        model_lib["train_tag"] + "_val_" + model_lib["features_type"],
        "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_pairs.list"
        )
    model_lib["val_speech_neg_pair_dir"] = path.join(
        pair_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"], 
        model_lib["train_tag"] + "_val_" + model_lib["features_type"],
        "key_" + model_lib["pair_type"] + "_negatives.list" if model_lib["pair_type"] in SPEECH_PAIR_TYPES else "key_negatives.list"
        )

    model_lib["speech_test_fn"] = path.join(
        feats_path, model_lib["speech_data_type"], "Subsets", "Words", model_lib["features_type"], 
        model_lib["train_tag"] + "_test_" + model_lib["features_type"] + ".npz"
        )

    model_lib["speech_episode_1_list"] = path.join(
        few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"], 1, model_lib["Q"], model_lib["speech_data_type"], "test"
            )
        ) 
    model_lib["speech_episode_5_list"] = path.join(few_shot_lib_path, episodes, "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
        model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["speech_data_type"], "test"
        )
    )
    

    model_lib["image_train_fn"] = path.join(feats_path, model_lib["image_data_type"], "train.npz")
    model_lib["image_pair_dir"] = path.join(pair_path, model_lib["image_data_type"], "train", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")
    model_lib["image_neg_pair_dir"] = path.join(pair_path, model_lib["image_data_type"], "train", "key_" + model_lib["pair_type"] + "_negatives.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_negatives.list")

    model_lib["image_val_fn"] = path.join(feats_path, model_lib["image_data_type"], "validation.npz")
    model_lib["val_image_pair_dir"] = path.join(pair_path, model_lib["image_data_type"], "validation", "key_" + model_lib["pair_type"] + "_pairs.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_pairs.list")
    model_lib["val_image_neg_pair_dir"] = path.join(pair_path, model_lib["image_data_type"], "validation", "key_" + model_lib["pair_type"] + "_negatives.list" if model_lib["pair_type"] in IMAGE_PAIR_TYPES else "key_negatives.list")

    model_lib["image_test_fn"] = path.join(feats_path, model_lib["image_data_type"], "test.npz")

    model_lib["image_episode_1_list"] = path.join(few_shot_lib_path, episodes, 
        "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"]-1, 1, model_lib["Q"], model_lib["image_data_type"], "test"
            )
        )
    model_lib["image_episode_5_list"] = path.join(few_shot_lib_path, episodes, 
        "M_{}_K_{}_Q_{}_{}_{}_episodes.txt".format(
            model_lib["M"]-1, model_lib["K"], model_lib["Q"], model_lib["image_data_type"], "test"
            )
        )

    model_lib["pair_dir"] = path.join(pair_path, "TIDigits_speech_MNIST_image_pairs", "train_pairs.txt")
    model_lib["val_pair_dir"] = path.join(pair_path, "TIDigits_speech_MNIST_image_pairs", "validation_pairs.txt")

    model_lib["episode_1_list"] = path.join(few_shot_lib_path, episodes, 
        "M_{}_K_{}_Q_{}_{}_{}_{}_{}_episodes.txt".format(
            model_lib["M"], 1, model_lib["Q"], model_lib["speech_data_type"], "test", model_lib["image_data_type"], "test"
            )
        )
    model_lib["episode_5_list"] = path.join(few_shot_lib_path, episodes, 
        "M_{}_K_{}_Q_{}_{}_{}_{}_{}_episodes.txt".format(
            model_lib["M"], model_lib["K"], model_lib["Q"], model_lib["speech_data_type"], "test", model_lib["image_data_type"], "test"
            )
        )

    return model_lib


def model_library_setup(base_lib, param_dict):
    model_lib = base_lib.copy()

    temp_list = []
    for pool_layer in model_lib["image_pool_layers"]:
        if pool_layer == "None": pool_value = None
        else: pool_value = pool_layer
        temp_list.append(pool_value)
    model_lib["image_pool_layers"] = temp_list

    for key in param_dict:
        if key != "rnd_seed":
            model_lib[key] = param_dict[key]

    model_lib["model_name"] = get_model_name(model_lib)

    date = str(datetime.now()).split(" ")
    model_lib["date"] = date
    model_lib["model_instance"] = model_lib["model_name"] + "_" + date[0] + "_" + date[1]

    model_lib["output_fn"] = path.join(model_path if model_lib["final_model"] else non_final_model_path, 
        model_lib["model_type"], model_lib["model_name"], model_lib["model_instance"]
        )
    util_library.check_dir(model_lib["output_fn"])

    model_lib["best_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_best")
    model_lib["intermediate_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_last")
    
    for key in param_dict:
        if key == "rnd_seed":
            model_lib[key] = param_dict[key]

    if model_lib["batch_size"] <= 0: 
        print("Batch size must be greater than zero")
        sys.exit(0)
 
    if model_lib["n_buckets"] <= 0 and model_lib["divide_into_buckets"]: 
        print("Number of buckets must be greater than zero")
        sys.exit(0)

    if model_lib["epochs"] <= 0: 
        print("Epochs must be greater than zero")
        sys.exit(0)

    if model_lib["learning_rate"] <= 0.0: 
        print("Learning rate must be greater than zero")
        sys.exit(0)

    if model_lib["keep_prob"] < 0.0 and model_lib["keep_prob"] > 1.0: 
        print("Keep probability must be between 0.0 and 1.0")
        sys.exit(0)
    
    log_dict, model_instances = model_files(model_lib)
    test_list = ["multimodal", "multimodal_zero_shot", "unimodal_image", "unimodal_speech"]
    exit_flag = True

    if str(model_lib["rnd_seed"]) in log_dict:
        print(f'This model with seed {model_lib["rnd_seed"]} already trained.')

        lib_fn = path.join("/".join(model_lib["output_fn"].split("/")[0:-1]), model_instances[str(model_lib["rnd_seed"])], model_lib["model_name"] + "_lib.pkl")
        model_lib = model_setup_library.restore_lib(lib_fn)
        for test in test_list:
            if test in log_dict[str(model_lib["rnd_seed"])]:
                model_lib["do_" + test] = False
            else:
                model_lib["do_" + test] = True
                exit_flag = False
        return False, exit_flag, model_lib
    else: 
        for test in test_list:
            model_lib["do_" + test] = True
        return True, False, model_lib

# def model_paths(model_lib, param_dict):

#     date = str(datetime.now()).split(" ")
#     model_lib["date"] = date
#     model_lib["model_instance"] = model_lib["model_name"] + "_" + date[0] + "_" + date[1]

#     model_lib["output_fn"] = path.join(model_path if model_lib["final_model"] else non_final_model_path, 
#         model_lib["model_type"], model_lib["model_name"], model_lib["model_instance"]
#         )
#     util_library.check_dir(model_lib["output_fn"])

#     model_lib["best_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_best")
#     model_lib["intermediate_model_fn"] = util_library.saving_path(model_lib["output_fn"], model_lib["model_name"] + "_" + model_lib["model_type"] + "_last")
    
#     for key in param_dict:
#         model_lib[key] = param_dict[key]
    
#     log_dict = model_files(model_lib)
    
#     if str(model_lib["rnd_seed"]) in log_dict:
#         print(f'This model with seed {model_lib["rnd_seed"]} already trained.')
#         model_setup_library.directory_management()
#         return True
#     else: return False

#_____________________________________________________________________________________________________________________________________
#
# Generating a model name
#
#_____________________________________________________________________________________________________________________________________

def get_model_name(lib):

    hasher = hashlib.md5(repr(sorted(lib.items())).encode("ascii"))
    
    return hasher.hexdigest()[:10]

#_____________________________________________________________________________________________________________________________________
#
# Model files and directories management
#
#_____________________________________________________________________________________________________________________________________

def model_files(lib):
    lib["unimodal_speech_log"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_unimodal_speech_log.txt"
        )
    lib["unimodal_image_log"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_unimodal_image_log.txt"
        )
    lib["multimodal_log"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_multimodal_log.txt"
        )
    lib["multimodal_zero_shot_log"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_multimodal_zero_shot_log.txt"
        )

    lib["model_lib_file"] = path.join(
        path.split(lib["output_fn"])[0], lib["model_name"] + "_library.txt"
        )

    if os.path.isfile(lib["unimodal_speech_log"]) is False:
        with open(lib["unimodal_speech_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()
    if os.path.isfile(lib["unimodal_image_log"]) is False:
        with open(lib["unimodal_image_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()
    if os.path.isfile(lib["multimodal_log"]) is False:
        with open(lib["multimodal_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()
    if os.path.isfile(lib["multimodal_zero_shot_log"]) is False:
        with open(lib["multimodal_zero_shot_log"], "w") as f:
            f.write(
                "Model name: {}\nLog file was created on {} at {}\n"
                .format(lib["model_name"], lib["date"][0], lib["date"][1])
                )
            f.close()

    if os.path.isfile(lib["model_lib_file"]) is False:
        with open(lib["model_lib_file"], "w") as f:
            f.write(
                "Model library:\n"
                )
            f.write("{")
            for key in sorted(lib):
                f.write("\t{}: {}\n".format(key, lib[key]))
            f.write("}\n")
            f.close()

    model_instances = {}
    unimodal_speech_log_dict = {}
    if os.path.isfile(lib["unimodal_speech_log"]):
        for line in open(lib["unimodal_speech_log"], 'r'):
            keyword = "rnd_seed of "
            if re.search(keyword, line):
                line_parts = line.strip().split(" ")
                keyword_parts = keyword.strip().split(" ")
                ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
                rnd_seed_1 = line_parts[ind[0]+2]
                acc_1 = line_parts[ind[0]-2]
                rnd_seed_few = line_parts[ind[1]+2]
                acc_few = line_parts[ind[1]-2]
                
                if rnd_seed_1 == rnd_seed_few and rnd_seed_1 not in unimodal_speech_log_dict:
                    unimodal_speech_log_dict[rnd_seed_1] = [acc_1, acc_few]
                    model_instances[rnd_seed_1] = ":".join(line_parts[0].split(":")[0:-1])

    unimodal_image_log_dict = {}
    if os.path.isfile(lib["unimodal_image_log"]):
        for line in open(lib["unimodal_image_log"], 'r'):
            keyword = "rnd_seed of "
            if re.search(keyword, line):
                line_parts = line.strip().split(" ")
                keyword_parts = keyword.strip().split(" ")
                ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
                rnd_seed_1 = line_parts[ind[0]+2]
                acc_1 = line_parts[ind[0]-2]
                rnd_seed_few = line_parts[ind[1]+2]
                acc_few = line_parts[ind[1]-2]
                
                if rnd_seed_1 == rnd_seed_few and rnd_seed_1 not in unimodal_image_log_dict:
                    unimodal_image_log_dict[rnd_seed_1] = [acc_1, acc_few]
                    if rnd_seed_1 not in model_instances:
                        model_instances[rnd_seed_1] = ":".join(line_parts[0].split(":")[0:-1])
                    elif model_instances[rnd_seed_1] != ":".join(line_parts[0].split(":")[0:-1]):
                        print("Something weird is going on.")
                        sys.exit(1)


    multimodal_log_dict = {}
    if os.path.isfile(lib["multimodal_log"]):
        for line in open(lib["multimodal_log"], 'r'):
            keyword = "rnd_seed of "
            if re.search(keyword, line):
                line_parts = line.strip().split(" ")
                keyword_parts = keyword.strip().split(" ")
                ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
                rnd_seed_1 = line_parts[ind[0]+2]
                acc_1 = line_parts[ind[0]-2]
                rnd_seed_few = line_parts[ind[1]+2]
                acc_few = line_parts[ind[1]-2]
                
                if rnd_seed_1 == rnd_seed_few and rnd_seed_1 not in multimodal_log_dict:
                    multimodal_log_dict[rnd_seed_1] = [acc_1, acc_few]
                    if rnd_seed_1 not in model_instances:
                        model_instances[rnd_seed_1] = ":".join(line_parts[0].split(":")[0:-1])
                    elif model_instances[rnd_seed_1] != ":".join(line_parts[0].split(":")[0:-1]):
                        print("Something weird is going on.")
                        sys.exit(1)

    multimodal_zero_shot_log_dict = {}
    if os.path.isfile(lib["multimodal_zero_shot_log"]):
        for line in open(lib["multimodal_zero_shot_log"], 'r'):
            keyword = "rnd_seed of "
            if re.search(keyword, line):
                line_parts = line.strip().split(" ")
                keyword_parts = keyword.strip().split(" ")
                ind = np.where(np.asarray(line_parts) == keyword_parts[0])[0]
                rnd_seed_1 = line_parts[ind[0]+2]
                acc_1 = line_parts[ind[0]-2]
                
                if rnd_seed_1 not in multimodal_zero_shot_log_dict:
                    multimodal_zero_shot_log_dict[rnd_seed_1] = [acc_1]
                    if rnd_seed_1 not in model_instances:
                        model_instances[rnd_seed_1] = ":".join(line_parts[0].split(":")[0:-1])
                    elif model_instances[rnd_seed_1] != ":".join(line_parts[0].split(":")[0:-1]):
                        print("Something weird is going on.")
                        sys.exit(1)

    log_dict = {}
    max_len = max(len(multimodal_log_dict.keys()), len(multimodal_zero_shot_log_dict.keys()), len(unimodal_image_log_dict.keys()), len(unimodal_speech_log_dict.keys()))
    len_dict = [len(multimodal_log_dict.keys()), len(multimodal_zero_shot_log_dict.keys()), len(unimodal_image_log_dict.keys()), len(unimodal_speech_log_dict.keys())]
    which_dict = np.argsort(len_dict)[::-1]
    index_dict = {
        0: "multimodal",
        1: "multimodal_zero_shot",
        2: "unimodal_image",
        3: "unimodal_speech"

    }
    dict_dict = {
        "multimodal": multimodal_log_dict, 
        "multimodal_zero_shot": multimodal_zero_shot_log_dict, 
        "unimodal_image": unimodal_image_log_dict, 
        "unimodal_speech": unimodal_speech_log_dict}


    for key in dict_dict[index_dict[which_dict[0]]]:

        for dict_key in dict_dict:
            this_dict = dict_dict[dict_key]
            if key in this_dict:
                if len(this_dict[key]) == 2 or (dict_key == "multimodal_zero_shot" and len(this_dict[key]) == 1):
                    if key not in log_dict:
                        log_dict[key] = {}
                    log_dict[key][dict_key] = this_dict[key]

    return log_dict, model_instances