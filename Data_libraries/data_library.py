#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
# Some fragment of code adapted from and credit given to: Herman Kamper
#_________________________________________________________________________________________________
#
# This script contains various functions to read in data from .npz files, to generate pairs, to 
# read in pairs from text files and truncate data.

from datetime import datetime
from os import path
import argparse
import glob
import numpy as np
import os
from os import path
from scipy.io import wavfile
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import subprocess
import sys
from tqdm import tqdm
from scipy.fftpack import dct
import random

sys.path.append("..")
from paths import model_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library

COL_LENGTH = model_setup_library.COL_LENGTH

#_____________________________________________________________________________________________________________________________________
#
# Data pairs
#
#_____________________________________________________________________________________________________________________________________

def data_pairs_from_file(file_fn, keys, add_both_directions=False):

    pair_list = []
    pair_set= set()
    pair_list = []

    for line in tqdm(open(file_fn, 'r'), desc="\tReading in pairs", ncols=COL_LENGTH):

        pairs = line.split()
        cur_key = np.where(np.asarray(keys) == pairs[0])[0][0]

        pair_key = np.where(np.asarray(keys) == pairs[1])[0][0]
        pair_set.add((cur_key, pair_key))
        if add_both_directions:
            pair_set.add((pair_key, cur_key))  
    

    for pair1, pair2 in pair_set:
        pair_list.append((pair1, pair2))

    return pair_list

def image_data_pairs(image_x, keys, labels):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(len(labels)), desc="\tGenerating pairs", ncols=COL_LENGTH):
        cur_label = labels[i]

        matching_labels = np.where(np.asarray(labels)== labels[i])[0]
        
        if len(matching_labels) > 0:
            count = 0
            pair = -1
            while pair == -1 and count < len(matching_labels):
                lower_limit = 0.05
                upper_limit = 0.25
                this_distance = cdist(image_x[i], image_x[matching_labels[count]], "cosine")

                if keys[i] != keys[matching_labels[count]] and this_distance > lower_limit and this_distance < upper_limit:
                    pair = matching_labels[count]
                count += 1
            
        if pair != -1: pair_list.append((i, pair))

    return pair_list

def data_pairs_from_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(len(labels)), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):

        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels)== labels[i])[0]
        
        if len(matching_labels) > 0:
            count = 0
            pair = -1
            
            while pair == -1 and count < len(matching_labels):
                speaker2 = keys[matching_labels[count]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair = matching_labels[count] 
                    break
                count += 1
            if pair != -1: pair_list.append((i, pair))
                       
    return pair_list

def speech_data_pairs_from_file(file_fn, keys, add_both_directions=False):

    pair_list = []
    pair_set= set()
    pair_list = []

    for line in tqdm(open(file_fn, 'r'), desc="\tReading in pairs", ncols=COL_LENGTH):

        pairs = line.split()
        cur_key = np.where(np.asarray(keys) == pairs[0])[0][0]

        # for i in range(1, len(pairs)):
        same_speaker_pair_key = np.where(np.asarray(keys) == pairs[1])[0][0]
        pair_set.add((cur_key, same_speaker_pair_key))
        if add_both_directions:
            pair_set.add((same_speaker_pair_key, cur_key))  

        different_speaker_pair_key = np.where(np.asarray(keys) == pairs[2])[0][0]
        pair_set.add((cur_key, different_speaker_pair_key))
        if add_both_directions:
            pair_set.add((different_speaker_pair_key, cur_key)) 
    

    for pair1, pair2 in pair_set:
        pair_list.append((pair1, pair2))

    return pair_list

def data_pairs(labels, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:
            pair_list.append((i, matching_labels[0]))
            if add_both_directions:
                pair_list.append((matching_labels[0], i))  

    return pair_list

def data_pairs_from_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:

            for j in range(len(matching_labels)):
                speaker2 = keys[matching_labels[j]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair_list.append((i, matching_labels[j]))
                    if add_both_directions:
                        pair_list.append((matching_labels[j], i))   
                    break
                       
    return pair_list

def data_pairs_from_same_and_different_speakers(labels, keys, add_both_directions=False):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenerating pairs from different speakers", ncols=COL_LENGTH):
        offset = i + 1
        cur_label = labels[i]
        speaker1 = keys[i].split("_")[1].split("-")[0]
        matching_labels = np.where(np.asarray(labels[i + 1:])== labels[i])[0] + offset
        if len(matching_labels) > 0:

            for j in range(len(matching_labels)):
                speaker2 = keys[matching_labels[j]].split("_")[1].split("-")[0]

                if speaker1 != speaker2:
                    pair_list.append((i, matching_labels[j]))
                    if add_both_directions:
                        pair_list.append((matching_labels[j], i))   
                    break
                       
    return pair_list

def all_data_pairs(labels, add_both_directions=True):

    N = len(labels)
    pair_list = []

    for i in tqdm(range(N-1), desc="\tGenrating all possible pairs", ncols=COL_LENGTH):
        offset = i + 1
        for matching_label_i in (np.where(np.asarray(labels[i + 1:]) == labels[i])[0] + offset):
            pair_list.append((i, matching_label_i))
            if add_both_directions:
                pair_list.append((matching_label_i, i))         
    return pair_list

def speech_image_pairs(speech_labels, speech_pair_list, image_labels, image_pair_list):
    
    N = min(len(image_pair_list), len(speech_pair_list))
    pair_count = 0
    pair_list = []
    for i in tqdm(range(len(speech_pair_list)), desc="\tGenerating pairs", ncols=COL_LENGTH):
        (pair_1, pair_2) = speech_pair_list[i]
        cur_label = speech_labels[pair_1]
        if cur_label == "z": search_key = "0"
        elif cur_label == "o": search_key = "0"
        else: search_key = cur_label
        image_indices = set(np.where(np.asarray(image_labels) == search_key)[0])
        image_pair_list_set = set([x for tup in image_pair_list for x in tup])
        valid_indices = list(image_indices.intersection(image_pair_list_set))
        search_for_pair = True
        ind_count = 0   

        while search_for_pair and len(valid_indices) > 0:
            if ind_count > len(valid_indices) - 1:
                search_for_pair = False
                break
            

            pair_ind = np.where(np.asarray(image_pair_list) == valid_indices[ind_count])[0]
            ind_count += 1
            pair_ind = pair_ind[0]
            (pair_3, pair_4) = image_pair_list[pair_ind]

            if pair_count < N:
                pair_list.append((pair_1, pair_2, pair_3, pair_4))
                pair_count += 1
                image_labels[pair_3] = "n"
                image_labels[pair_4] = "n"

                search_for_pair = False


    return pair_list

def speech_image_siamese_like_pairs(speech_labels, speech_keys, image_x, image_labels):
    
  N = min(len(image_labels), len(speech_labels))
  pair_count = 0
  pop_count = 0

  pair_list = []

  all_speech_indices = [k for k in range(len(speech_labels))]
  all_image_indices = [k for k in range(len(image_labels))]
  speaker_list = [key.split("_")[1].split("-")[0] for key in speech_keys]

  speaker_set = set(speaker_list)
  all_speech_labels = set([key for key in speech_labels])
  all_image_labels = set([key for key in image_labels])
  
  speech_dict = {}
  speaker_dict = {}
  pos_image_dict = {}
  neg_image_dict = {}
  
  for this_label in all_speech_labels:
    speech_indices = np.where(np.asarray(speech_labels) == this_label)[0]
    speech_dict[this_label] = list(set(all_speech_indices).difference(set(speech_indices)))

  for this_speaker in speaker_set:
    speaker_dict[this_speaker] = list(np.where(np.asarray(speaker_list) == this_speaker)[0])

  for this_label in all_image_labels:
    image_indices = np.where(np.asarray(image_labels) == this_label)[0]
    pos_image_dict[this_label] = list(image_indices)
    valid_image_indices = set(all_image_indices).difference(set(image_indices))
    neg_image_dict[this_label] = list(valid_image_indices)

  for i in tqdm(all_speech_indices, desc="\tGenerating pairs", ncols=COL_LENGTH):
    sp_pos = i
    cur_label = speech_labels[sp_pos]
    cur_speaker = speech_keys[sp_pos].split("_")[1].split("-")[0]

    if cur_label == "z": search_key = "0"
    elif cur_label == "o": search_key = "0"
    else: search_key = cur_label

    search_for_pair = True

    while search_for_pair and pair_count < N:
      sp_neg = -1
      im_pos = -1
      im_neg = -1
      
      if len(speech_dict[cur_label]) != 0 and len(speaker_dict[cur_speaker]) != 0 and len(pos_image_dict[search_key]) != 0 and len(neg_image_dict[search_key]) != 0:
          
        possible_sp_neg = list(set(speech_dict[cur_label]).intersection(set(speaker_dict[cur_speaker])))
        
        if len(possible_sp_neg) != 0:
          sp_neg = possible_sp_neg[0]
          im_pos = pos_image_dict[search_key][0]
          
          while im_neg == -1 and len(neg_image_dict[search_key]) > 0:
            im_count = random.randint(0, len(neg_image_dict[search_key]) - 1)
            possible_im_neg = neg_image_dict[search_key][im_count]

            this_distance = cdist(image_x[im_pos], image_x[possible_im_neg], "cosine")
            lower_limit = 0.6
            upper_limit = 0.8
            if this_distance > lower_limit and this_distance < upper_limit: 
              im_neg = possible_im_neg

          if sp_neg != -1 and im_pos != -1 and im_neg != -1:
            speech_dict[cur_label].remove(sp_neg)
            speaker_dict[cur_speaker].remove(sp_neg)
            pos_image_dict[search_key].remove(im_pos)
            neg_image_dict[search_key].remove(im_neg)

            pair_list.append((sp_pos, sp_neg, im_pos, im_neg))
            pair_count += 1
          
          search_for_pair = False  

        else: search_for_pair = False
      else: search_for_pair = False

  return pair_list



def speech_image_pairs_from_file(file_fn, speech_keys, speech_pair_list, image_keys, image_pair_list):

  pair_list = []
  for line in tqdm(open(file_fn, 'r'), desc="\tGenerating pairs", ncols=COL_LENGTH):
    (pair_1, pair_3) = line.split()
    pair_1 = np.where(np.asarray(speech_keys) == pair_1)[0][0]
    pair_3 = np.where(np.asarray(image_keys) == pair_3)[0][0]

    ind_search = np.where((np.asarray(speech_pair_list) == pair_1))
    ind_ind = np.where(ind_search[1] == 0)[0]
    if len(ind_search[0]) != 0 and len(ind_ind) != 0:
      pair_list_ind = ind_search[0][ind_ind[0]]
      pair_list_entries = speech_pair_list[pair_list_ind]
      pair_2 = pair_list_entries[1]
    else:
      pair_2 = -1

    if pair_2 != -1:
      ind_search = np.where((np.asarray(image_pair_list) == pair_3))
      ind_ind = np.where(ind_search[1] == 0)[0]
      if len(ind_search[0]) != 0 and len(ind_ind) != 0:
        pair_list_ind = ind_search[0][ind_ind[0]]
        pair_list_entries = image_pair_list[pair_list_ind]
        pair_4 = pair_list_entries[1]
      else:
        pair_4 = -1

    if pair_2 != -1 and pair_4 != -1: 
      pair_list.append((pair_1, pair_2, pair_3, pair_4))

  return pair_list

def speech_image_siamese_like_pairs_from_file(
    file_fn, speech_keys, speech_neg_pair_list, image_keys, image_neg_pair_list):
    pair_list = []
    for line in tqdm(open(file_fn, 'r'), desc="\tGenerating pairs", ncols=COL_LENGTH):
        (pair_1, pair_3) = line.split()
        pair_1 = np.where(np.asarray(speech_keys) == pair_1)[0][0]
        pair_3 = np.where(np.asarray(image_keys) == pair_3)[0][0]
        
        ind_search = np.where((np.asarray(speech_neg_pair_list) == pair_1))
        ind_ind = np.where(ind_search[1] == 0)[0]
        if len(ind_search[0]) == 2 and len(ind_ind) != 0:
          pair_list_ind = ind_search[0][ind_ind[0]]
          pair_list_entries = speech_neg_pair_list[pair_list_ind]
          pair_2 = pair_list_entries[1]
        else:
          pair_2 = -1
            
        if pair_2 != -1:
          ind_search = np.where((np.asarray(image_neg_pair_list) == pair_3))
          ind_ind = np.where(ind_search[1] == 0)[0]
          if len(ind_search[0]) != 0 and len(ind_ind) != 0:
            pair_list_ind = ind_search[0][ind_ind[0]]
            pair_list_entries = image_neg_pair_list[pair_list_ind]
            pair_4 = pair_list_entries[1]
          else:
            pair_4 = -1

        if pair_2 != -1 and pair_4 != -1: 
          pair_list.append((pair_1, pair_2, pair_3, pair_4))

    return pair_list

#_____________________________________________________________________________________________________________________________________
#
# Loading in data
#
#_____________________________________________________________________________________________________________________________________

def load_image_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    words = []
    keys = []
    n_items = 0
    for im_key in tqdm(sorted(npz), desc="\tExtracting image data from {}".format(fn), ncols=COL_LENGTH):
        keys.append(im_key)
        feats.append(npz[im_key])
        word = im_key.split("_")[0]
        words.append(word)
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample label of a feature: {}".format(words[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, words, keys)

def load_speech_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    words = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in tqdm(sorted(npz), desc="\tExtracting speech data from {}".format(fn), ncols=COL_LENGTH):
        keys.append(utt_key)
        feats.append(npz[utt_key])
        word = utt_key.split("_")[0]
        words.append(word)
        lengths.append(npz[utt_key].shape[0])
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample label of a feature: {}".format(words[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, words, lengths, keys)

def load_latent_data_from_npz(fn):
    npz = np.load(fn)

    feats = []
    keys = []
    n_items = 0
    for key in tqdm(sorted(npz), desc="\tExtracting latents from {}".format(fn), ncols=COL_LENGTH):
        keys.append(key)
        feats.append(npz[key])
        n_items += 1
        
    print("\tNumber of items extracted: {}".format(n_items))
    print("\tExample key of a feature: {}".format(keys[0]))
    print("\tExample shape of a feature: {}".format(feats[0].shape))
    return (feats, keys)

#_____________________________________________________________________________________________________________________________________
#
# Truncating and limiting speech data
#
#_____________________________________________________________________________________________________________________________________


def truncate_data_dim(feats, lengths, max_feat_dim, max_frames):
    print("\n\tLimiting dimensionality: {}".format(max_feat_dim))
    print("\tLimiting number of frames: {}".format(max_frames))
    for i in tqdm(range(len(feats)), desc="\tTruncating", ncols=COL_LENGTH):
        feats[i] = feats[i][:max_frames, :max_feat_dim]
        lengths[i] = min(lengths[i], max_frames)

    
#_____________________________________________________________________________________________________________________________________
#
# Zeropadding speech data
#
#_____________________________________________________________________________________________________________________________________

def pad_speech_data(input_x, pad_to_num, return_mask=False):

    padded_data = np.zeros((len(input_x), pad_to_num, input_x[0].shape[1]), dtype=np.float32)
    lengths = []
    if return_mask: mask = np.zeros((len(input_x), pad_to_num), dtype=np.float32)

    for i, data in tqdm(enumerate(input_x), desc="\tPadding", ncols=COL_LENGTH):

        data_length = data.shape[0]
        padding = int((pad_to_num - data_length)/2)
        
        if data_length <= pad_to_num:
            padded_data[i, padding:padding+data_length, :] = data
            lengths.append(min(data_length, pad_to_num))
            if return_mask: mask[i, padding:padding+data_length] = 1

        else: 
            data_length = min(data_length, pad_to_num)
            padded_data[i, :data_length, :] = data[-padding:-padding+pad_to_num]
            lengths.append(data_length)
            if return_mask: mask[i, :] = 1

    if return_mask: return (padded_data, lengths, mask)
    else: return (padded_data, lengths)

#_____________________________________________________________________________________________________________________________________
#
# Getting a mask for speech data
#
#_____________________________________________________________________________________________________________________________________

def get_mask(input_x, input_lengths, max_length=None):
    if max_length is None: mask = np.zeros((len(input_x), max(input_lengths)), dtype=np.float32)
    else: mask = np.zeros((len(input_x), max_length), dtype=np.float32)

    for i, data in tqdm(enumerate(input_x), desc="\tGenerating mask", ncols=COL_LENGTH):

        data_length = data.shape[0]
        
        if data_length <= max(input_lengths): mask[i, 0:data_length] = 1
        elif max_length is not None and data_length <= max_length: mask[i, 0:data_length] = 1
        else: mask[i, :] = 1

    return mask


def flatten_speech_data(input_x, input_dim):

    input_x = np.transpose(input_x, (0, 2, 1))
    input_x = input_x.reshape((-1, input_dim))
    return input_x


def remove_test_classes(x, labels, lengths, keys, lab_to_exclude):
    speech_x = []
    speech_labels = []
    speech_lengths = []
    speech_keys = []
    for i, label in tqdm(enumerate(labels), desc="\tRemoving test classes", ncols=COL_LENGTH):
        if label not in lab_to_exclude:
            speech_x.append(x[i])
            speech_labels.append(labels[i])
            speech_lengths.append(lengths[i])
            speech_keys.append(keys[i])
    return (speech_x.copy(), speech_labels.copy(), speech_lengths.copy(), speech_keys.copy())

def test_classes(labels, lab_to_exclude, label_type):

    test_labels_present = False
    for label in tqdm(labels, desc="\tTesting {} labels".format(label_type), ncols=COL_LENGTH):
        if label in lab_to_exclude:
            test_labels_present = True

    if test_labels_present:
        print("\tTest classes present in {} dataset".format(label_type))