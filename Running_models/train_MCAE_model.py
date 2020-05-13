#!/usr/bin/env python

#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script trains a MCAE model from a library.
#

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import argparse
from IPython import display
import timeit
import re
import random
import json
import sys
import os
from os import path
from tqdm import tqdm


sys.path.append("..")
from paths import data_lib_path
from paths import model_lib_path
from paths import few_shot_lib_path

sys.path.append(path.join("..", model_lib_path))
import model_setup_library
import MCAE_and_MTriplet_setup_library
import model_legos_library
import MCAE_and_MTriplet_model_legos_library
import training_library
model_lib_path = path.join("..", model_lib_path)

sys.path.append(path.join("..", data_lib_path))
import data_library
import batching_library

sys.path.append(path.join("..", few_shot_lib_path))
import few_shot_learning_library
import generate_unimodal_image_episodes
import generate_unimodal_speech_episodes
import generate_episodes

PRINT_LENGTH =  180
COL_LENGTH =  PRINT_LENGTH - len("\t".expandtabs())


import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#_____________________________________________________________________________________________________________________________________
#
# Reading in command line arguments
#
#_____________________________________________________________________________________________________________________________________

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_seeds", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--test_batch_size", type=str, choices=["True", "False"], default="True")
    return parser.parse_args()

parameters = arguments()

test_seeds = parameters.test_seeds == "True"
if test_seeds:
  BATCH_SIZES = [16, 32, 64, 128, 256, 512]
else:
  BATCH_SIZES = [256]

test_batch_size = parameters.test_batch_size == "True"
if test_batch_size:
  SEEDS = [1, 5, 10, 21, 42]
else:
  SEEDS = [1]

#_____________________________________________________________________________________________________________________________________
#
# Setting up a model library
#
#_____________________________________________________________________________________________________________________________________

with open(path.join(model_lib_path, "config.json")) as file:
  base_lib = json.load(file)

base_lib = MCAE_and_MTriplet_setup_library.base_library_setup(base_lib)

#_____________________________________________________________________________________________________________________________________
#
# Data procesing
#
#_____________________________________________________________________________________________________________________________________

print("\n" + "-"*PRINT_LENGTH)
print("Processing training data")
print("-"*PRINT_LENGTH)
image_train_x, image_train_labels, image_train_keys = (
  data_library.load_image_data_from_npz(base_lib["image_train_fn"])
  )
speech_train_x, speech_train_labels, speech_train_lengths, speech_train_keys = (
  data_library.load_speech_data_from_npz(base_lib["speech_train_fn"])
  )
data_library.truncate_data_dim(speech_train_x, speech_train_lengths, base_lib["speech_input_dim"], base_lib["max_frames"])
train_mask = data_library.get_mask(speech_train_x, speech_train_lengths, None)
print("\n" + "-"*PRINT_LENGTH)
print("Processing validation data")
print("-"*PRINT_LENGTH)
image_val_x, image_val_labels, image_val_keys = (
  data_library.load_image_data_from_npz(base_lib["image_val_fn"])
  )
speech_val_x, speech_val_labels, speech_val_lengths, speech_val_keys = (
  data_library.load_speech_data_from_npz(base_lib["speech_val_fn"])
  )
data_library.truncate_data_dim(speech_val_x, speech_val_lengths, base_lib["speech_input_dim"], base_lib["max_frames"])
val_mask = data_library.get_mask(speech_val_x, speech_val_lengths)
print("\n" + "-"*PRINT_LENGTH)
print("Processing test data")
print("-"*PRINT_LENGTH)
image_test_x, image_test_labels, image_test_keys = (
    data_library.load_image_data_from_npz(base_lib["image_test_fn"])
    )
speech_test_x, speech_test_labels, speech_test_lengths, speech_test_keys = (
    data_library.load_speech_data_from_npz(base_lib["speech_test_fn"])
    )
data_library.truncate_data_dim(speech_test_x, speech_test_lengths, base_lib["speech_input_dim"], base_lib["max_frames"])

#_____________________________________________________________________________________________________________________________________
#
# Generating speech-image pairs
#
#_____________________________________________________________________________________________________________________________________

print("\n" + "-"*PRINT_LENGTH)
print("Generating training pairs")
print("-"*PRINT_LENGTH)
if base_lib["topline"]:
  speech_pair_list = data_library.data_pairs_from_different_speakers(speech_train_labels, speech_train_keys)
  image_pair_list = data_library.image_data_pairs(image_train_x, image_train_keys, image_train_labels)
  speech_image_pair_list = data_library.speech_image_pairs(speech_train_labels, speech_pair_list, image_train_labels.copy(), image_pair_list.copy())
else:
  speech_pair_list = data_library.speech_data_pairs_from_file(base_lib["speech_pair_dir"], speech_train_keys)
  image_pair_list = data_library.data_pairs_from_file(base_lib["image_pair_dir"], image_train_keys)
  speech_image_pair_list = data_library.speech_image_pairs_from_file(base_lib["pair_dir"], speech_train_keys, speech_pair_list, image_train_keys, image_pair_list)

print("\n" + "-"*PRINT_LENGTH)
print("Generating validation pairs")
print("-"*PRINT_LENGTH)
if base_lib["topline"]:
  val_speech_pair_list = data_library.data_pairs_from_different_speakers(speech_val_labels, speech_val_keys)
  val_image_pair_list = data_library.image_data_pairs(image_val_x, image_val_keys, image_val_labels)
  val_speech_image_pair_list = data_library.speech_image_pairs(speech_val_labels, val_speech_pair_list, image_val_labels.copy(), val_image_pair_list.copy())
else:
  val_speech_pair_list = data_library.speech_data_pairs_from_file(base_lib["val_speech_pair_dir"], speech_val_keys)
  val_image_pair_list = data_library.data_pairs_from_file(base_lib["val_image_pair_dir"], image_val_keys)
  val_speech_image_pair_list = data_library.speech_image_pairs_from_file(base_lib["val_pair_dir"], speech_val_keys, val_speech_pair_list, image_val_keys, val_image_pair_list)

#_____________________________________________________________________________________________________________________________________
#
# Training models at different batch_sizes
#
#_____________________________________________________________________________________________________________________________________

for batch_size in BATCH_SIZES:
  #_____________________________________________________________________________________________________________________________________
  #
  # Training model with certain batch_size with different seeds
  #
  #_____________________________________________________________________________________________________________________________________

  for rnd_seed in SEEDS:
    print("\n" + "-"*PRINT_LENGTH)
    print("Training model instance")
    print("-"*PRINT_LENGTH)

    #_____________________________________________________________________________________________________________________________________
    #
    # Updating the model library and determining whether training and which tests should be done 
    #
    #_____________________________________________________________________________________________________________________________________

    temp_lib = {
        "batch_size": batch_size,
        "rnd_seed": rnd_seed
      }
    train, exit_flag, lib = MCAE_and_MTriplet_setup_library.model_library_setup(base_lib, temp_lib)

    if exit_flag is False:

      model_setup_library.lib_print(lib)
      np.random.seed(lib["rnd_seed"])
      tf.set_random_seed(lib["rnd_seed"])
      tf.reset_default_graph()

      #_____________________________________________________________________________________________________________________________________
      #
      # Data iterators
      #
      #_____________________________________________________________________________________________________________________________________

      train_it = batching_library.bucketing_pair_speech_and_image_iterator(speech_train_x, image_train_x,
              speech_image_pair_list, lib["batch_size"], lib["n_buckets"],
              shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"], 
              mask_x=train_mask, return_mask=True)
      val_it = batching_library.bucketing_pair_speech_and_image_iterator(speech_val_x, image_val_x,
              val_speech_image_pair_list, lib["batch_size"], lib["n_buckets"],
              shuffle_batches_every_epoch=lib["shuffle_batches_every_epoch"], 
              mask_x=val_mask, return_mask=True)

      #_____________________________________________________________________________________________________________________________________
      #
      # Model setup
      #
      #_____________________________________________________________________________________________________________________________________
      
      print("\n" + "-"*PRINT_LENGTH)
      print("Speech-part model setup")
      print("-"*PRINT_LENGTH)
      speech_X = tf.placeholder(tf.float32, [None, None, lib["speech_input_dim"]])
      speech_X_lengths = tf.placeholder(tf.int32, [None])
      speech_target = tf.placeholder(tf.float32, [None, None, lib["speech_input_dim"]])
      target_lengths = tf.placeholder(tf.int32, [None])
      X_mask = tf.placeholder(tf.float32, [None, None])
      target_mask = tf.placeholder(tf.float32, [None, None])

      train_flag = tf.placeholder_with_default(False, shape=())

      speech_model = MCAE_and_MTriplet_model_legos_library.rnn_architecture([speech_X, speech_X_lengths], train_flag, model_setup_library.activation_lib(), lib, target_lengths, True)
      speech_latent = speech_model["latent"]
      speech_output = speech_model["output"]
      masked_output = speech_output * tf.expand_dims(target_mask, -1)

      print("\n" + "-"*PRINT_LENGTH)
      print("Image-part model setup")
      print("-"*PRINT_LENGTH)

      image_X = tf.placeholder(tf.float32, [None, 28, 28, 1])
      image_target = tf.placeholder(tf.float32, [None, 28, 28, 1])

      image_model = MCAE_and_MTriplet_model_legos_library.cnn_architecture(image_X, train_flag, lib["image_enc"], lib["image_enc_strides"], model_setup_library.pooling_lib(), 
        lib["image_pool_layers"], lib["latent"], lib["image_dec"], lib["image_dec_strides"], lib, model_setup_library.activation_lib(), True)
      image_latent = image_model["latent"]
      image_output = image_model["output"]

      training_placeholders = [image_X, image_target, speech_X, speech_X_lengths, X_mask, speech_target, target_lengths, target_mask]

      loss = (
        0.3 * tf.reduce_mean(
          tf.reduce_sum(
            tf.reduce_mean(tf.square(speech_target-masked_output), -1), -1
            )/ tf.reduce_sum(target_mask, 1)
          ) 
        + 0.3 * tf.reduce_mean(tf.pow(image_target - image_output, 2))
        + 0.4 * tf.reduce_mean(tf.pow(speech_latent - image_latent, 2))
        )
      optimization = tf.train.AdamOptimizer(lib["learning_rate"]).minimize(loss)

      if train:
        
        #_____________________________________________________________________________________________________________________________________
        #
        # Training model
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Training model")
        print("-"*PRINT_LENGTH)
        record, train_log = training_library.training_model(
            [loss, optimization, train_flag], training_placeholders, lib, train_it,
            lib["epochs"], lib["patience"], lib["min_number_epochs"], lib["model_type"], val_it, None,
                None, save_model_fn=lib["intermediate_model_fn"], save_best_model_fn=lib["best_model_fn"]
                )
        model_setup_library.save_lib(lib)

      model_fn = model_setup_library.get_model_fn(lib)

      if lib["do_unimodal_speech"]:
        #_____________________________________________________________________________________________________________________________________
        #
        # Unimodal speech test
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Unimodal speech classification tests")
        print("-"*PRINT_LENGTH)
        def unimodal_speech_test(
          episode_fn=lib['speech_episode_1_list'], data_x=speech_test_x, data_keys=speech_test_keys, 
          data_labels=speech_test_labels, normalize=False, k=1
          ):
                episode_dict = generate_unimodal_speech_episodes.read_in_episodes(episode_fn)

                correct = 0
                total = 0

                episode_numbers = np.arange(1, len(episode_dict)+1)
                np.random.shuffle(episode_numbers)

                saver = tf.train.Saver()
                with tf.Session() as sesh:
                    saver.restore(sesh, model_fn)

                    for episode in tqdm(episode_numbers, desc=f'\t{k}-shot unimodal speech classification tests on {len(episode_numbers)} episodes for random seed {rnd_seed:2.0f}', ncols=COL_LENGTH):

                        episode_num = str(episode)
                        query = episode_dict[episode_num]["query"]
                        query_data, query_keys, query_lab = generate_unimodal_speech_episodes.episode_data(
                            query["keys"], data_x, data_keys, data_labels
                            )
                        query_iterator = batching_library.speech_iterator(
                            query_data, len(query_data), shuffle_batches_every_epoch=False
                            )
                        query_labels = [query_lab[i] for i in query_iterator.indices]

                        support_set = episode_dict[episode_num]["support_set"]
                        S_data, S_keys, S_lab = generate_unimodal_speech_episodes.episode_data(
                            support_set["keys"], data_x, data_keys, data_labels
                            )
                        S_iterator = batching_library.speech_iterator(
                            S_data, len(S_data), shuffle_batches_every_epoch=False
                            )
                        S_labels = [S_lab[i] for i in S_iterator.indices]


                        for feats, lengths in query_iterator:
                            lat = sesh.run([speech_latent], feed_dict={speech_X: feats, speech_X_lengths: lengths, train_flag: False})[0]

                        for feats, lengths in S_iterator:
                            S_lat = sesh.run([speech_latent], feed_dict={speech_X: feats, speech_X_lengths: lengths, train_flag: False})[0]

                        if normalize: 
                            latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                            s_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
                        else: 
                            latents = lat
                            s_latents = S_lat

                        distances = cdist(latents, s_latents, "cosine")
                        
                        indexes = np.argmin(distances, axis=1)
                        label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_labels, S_labels)

                        for i in range(len(indexes)):
                            total += 1
                            if label_matches[i, indexes[i]]:
                                correct += 1

                return correct/total

        log = ""
        acc = unimodal_speech_test(
          lib["speech_episode_1_list"], speech_test_x, speech_test_keys, speech_test_labels, 
          normalize=True, k=1
          )
        print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
        log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])

        acc = unimodal_speech_test(
          lib["speech_episode_5_list"], speech_test_x, speech_test_keys, speech_test_labels, 
          normalize=True, k=5
          )
        print(f'\tAccuracy of {5}-shot task: {acc*100:.2f}%')

        log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])
        print("\tWriting: {}".format(lib["unimodal_speech_log"]))
        with open(lib["unimodal_speech_log"], "a") as write_results:
            write_results.write("\n{}: ".format(lib["model_instance"]) + log)


      if lib["do_unimodal_image"]:
        #_____________________________________________________________________________________________________________________________________
        #
        # Unimodal image test
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Unimodal image classification tests")
        print("-"*PRINT_LENGTH)
        def unimodal_image_test(
          episode_fn=lib["image_episode_1_list"], data_x=image_test_x, data_keys=image_test_keys, 
          data_labels=image_test_labels, normalize=True, k=1
          ):

                episode_dict = generate_unimodal_image_episodes.read_in_episodes(episode_fn)
                correct = 0
                total = 0

                episode_numbers = np.arange(1, len(episode_dict)+1)
                np.random.shuffle(episode_numbers)

                saver = tf.train.Saver()
                with tf.Session() as sesh:
                    saver.restore(sesh, model_fn)

                    for episode in tqdm(episode_numbers, desc=f'\t{k}-shot unimodal image classification tests on {len(episode_numbers)} episodes for random seed {rnd_seed:2.0f}', ncols=COL_LENGTH):

                        episode_num = str(episode)
                        query = episode_dict[episode_num]["query"]
                        query_data, query_keys, query_lab = generate_unimodal_image_episodes.episode_data(
                            query["keys"], data_x, data_keys, data_labels
                            )
                        query_iterator = batching_library.unflattened_image_iterator(
                            query_data, len(query_data), shuffle_batches_every_epoch=False
                            )
                        query_labels = [query_lab[i] for i in query_iterator.indices]

                        support_set = episode_dict[episode_num]["support_set"]
                        S_data, S_keys, S_lab = generate_unimodal_image_episodes.episode_data(
                            support_set["keys"], data_x, data_keys, data_labels
                            )
                        S_iterator = batching_library.unflattened_image_iterator(
                            S_data, len(S_data), shuffle_batches_every_epoch=False
                            )
                        S_labels = [S_lab[i] for i in S_iterator.indices]


                        for feats in query_iterator:
                            lat = sesh.run([image_latent], feed_dict={image_X: feats, train_flag: False})[0]

                        for feats in S_iterator:
                            S_lat = sesh.run([image_latent], feed_dict={image_X: feats, train_flag: False})[0]

                        if normalize: 
                            latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                            s_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
                        else: 
                            latents = lat
                            s_latents = S_lat

                        distances = cdist(latents, s_latents, "cosine")
                        indexes = np.argmin(distances, axis=1)
                        label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_labels, S_labels)

                        for i in range(len(indexes)):
                            total += 1
                            if label_matches[i, indexes[i]]:
                                correct += 1

                return correct/total

        log = ""
        acc = unimodal_image_test(
          lib["image_episode_1_list"], image_test_x, image_test_keys, image_test_labels, normalize=True, k=1
          )
        print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
        log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])

        acc = unimodal_image_test(
          lib["image_episode_5_list"], image_test_x, image_test_keys, image_test_labels, normalize=True, k=5
          )
        print(f'\tAccuracy of {5}-shot task: {acc*100:.2f}%')
        log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])

        print("\tWriting: {}".format(lib["unimodal_image_log"]))
        with open(lib["unimodal_image_log"], "a") as write_results:
            write_results.write("\n{}: ".format(lib["model_instance"]) + log)


      if lib["do_multimodal"]:
        #_____________________________________________________________________________________________________________________________________
        #
        # Multimodal test with support set
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Multimodal matching tests")
        print("-"*PRINT_LENGTH)
        def multimodal_task(
          episode_fn, sp_test_x, sp_test_keys, sp_test_labels, im_test_x, im_test_keys, im_test_labels, normalize=True, k=1
          ):

          episode_dict = generate_episodes.read_in_episodes(episode_fn)
          np.random.seed(rnd_seed)
          episode_numbers = np.arange(1, len(episode_dict)+1)
          np.random.shuffle(episode_numbers)
          correct = 0
          total = 0
          
          saver = tf.train.Saver()
          with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in tqdm(episode_numbers, desc=f'\t{k}-shot multimodal matching tests on {len(episode_numbers)} episodes for random seed {rnd_seed:2.0f}', ncols=COL_LENGTH):
            
              episode_num = str(episode)
              # Get query iterator
              query = episode_dict[episode_num]["query"]
              query_data, query_keys, query_lab = generate_episodes.episode_data(query["keys"], sp_test_x, sp_test_keys, sp_test_labels)
              query_iterator = batching_library.speech_iterator(query_data, len(query_data), shuffle_batches_every_epoch=False)
              
              # Get speech_support set
              support_set = episode_dict[episode_num]["support_set"]
              S_image_data, S_image_keys, S_image_lab = generate_episodes.episode_data(support_set["image_keys"], im_test_x, im_test_keys, im_test_labels)
              S_speech_data, S_speech_keys, S_speech_lab = generate_episodes.episode_data(support_set["speech_keys"], sp_test_x, sp_test_keys, sp_test_labels)
              key_list = []
              for i in range(len(S_speech_keys)):
                key_list.append((S_speech_keys[i], S_image_keys[i]))
              
              S_speech_iterator = batching_library.speech_iterator(S_speech_data, len(S_speech_data), shuffle_batches_every_epoch=False)
              
              for feats, lengths in query_iterator: lat = sesh.run([speech_latent], feed_dict={speech_X: feats, speech_X_lengths: lengths, train_flag: False})[0]
              for feats, lengths in S_speech_iterator: S_lat = sesh.run([speech_latent], feed_dict={speech_X: feats, speech_X_lengths: lengths, train_flag: False})[0]
              
              if normalize:
                query_latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                s_speech_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
              else: 
                query_latents = lat
                s_speech_latents = S_lat
              
              query_latent_labels = [query_lab[i] for i in query_iterator.indices]
              query_latent_keys = [query_keys[i] for i in query_iterator.indices]
              s_speech_latent_labels = [S_speech_lab[i] for i in S_speech_iterator.indices]
              s_speech_latent_keys = [S_speech_keys[i] for i in S_speech_iterator.indices]
              
              distances1 = cdist(query_latents, s_speech_latents, "cosine")
              indexes1 = np.argmin(distances1, axis=1)
              
              chosen_speech_keys = []
              for i in range(len(indexes1)):
                chosen_speech_keys.append(s_speech_latent_keys[indexes1[i]])
                
              S_image_iterator = batching_library.unflattened_image_iterator(S_image_data, len(S_image_data), shuffle_batches_every_epoch=False)
              matching_set = episode_dict[episode_num]["matching_set"]
              M_data, M_keys, M_lab = generate_episodes.episode_data(matching_set["keys"], im_test_x, im_test_keys, im_test_labels)
              M_image_iterator = batching_library.unflattened_image_iterator(M_data, len(M_data), shuffle_batches_every_epoch=False)
              
              for feats in S_image_iterator: lat = sesh.run([image_latent], feed_dict={image_X: feats, train_flag: False})[0]
              for feats in M_image_iterator: S_lat = sesh.run([image_latent], feed_dict={image_X: feats, train_flag: False})[0]
              
              if normalize:
                s_image_latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                matching_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
              else: 
                s_image_latents = lat
                s_latents = S_lat
                
              s_image_latent_labels = [S_image_lab[i] for i in S_image_iterator.indices]
              s_image_latent_keys = [S_image_keys[i] for i in S_image_iterator.indices]
              
              matching_latent_labels = [M_lab[i] for i in M_image_iterator.indices]
              matching_latent_keys = [M_keys[i] for i in M_image_iterator.indices]
              
              image_key_order_list = [] 
              s_image_latents_in_order = np.empty((query_latents.shape[0], s_image_latents.shape[1]))
              s_image_labels_in_order = [] 
              
              for j, key in enumerate(chosen_speech_keys):
                for (sp_key, im_key) in key_list:
                  if key == sp_key:
                    image_key_order_list.append(im_key)
                    i = np.where(np.asarray(s_image_latent_keys) == im_key)[0][0]
                    s_image_latents_in_order[j:j+1, :] = s_image_latents[i:i+1, :]
                    s_image_labels_in_order.append(s_image_latent_labels[i])
                    break
              
              distances2 = cdist(s_image_latents_in_order, matching_latents, "cosine")
              indexes2 = np.argmin(distances2, axis=1)
              label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_latent_labels, matching_latent_labels)
              
              for i in range(len(indexes2)):
                total += 1
                if label_matches[i, indexes2[i]]:
                  correct += 1
                  
          return correct/total

        log = ""
        acc = multimodal_task(
          lib["episode_1_list"], speech_test_x, speech_test_keys, speech_test_labels, 
          image_test_x, image_test_keys, image_test_labels, normalize=True, k=1
          )
        print(f'\tAccuracy of {1}-shot task: {acc*100:.2f}%')
        log += "One-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])

        acc = multimodal_task(
          lib["episode_5_list"], speech_test_x, speech_test_keys, speech_test_labels, 
          image_test_x, image_test_keys, image_test_labels, normalize=True, k=5
          )
        print(f'\tAccuracy of {5}-shot task: {acc*100:.2f}%')
        log += "{}-shot accuracy of {} at rnd_seed of {} ".format(lib["K"], acc, lib["rnd_seed"])
        print("\tWriting: {}".format(lib["multimodal_log"]))
        with open(lib["multimodal_log"], "a") as write_results:
            write_results.write("\n{}: ".format(lib["model_instance"]) + log)

      if lib["do_multimodal_zero_shot"]:
        #_____________________________________________________________________________________________________________________________________
        #
        # Multimodal test without support set
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Multimodal matching tests")
        print("-"*PRINT_LENGTH)
        def zero_shot_multimodal_task(
          episode_fn, sp_test_x, sp_test_keys, sp_test_labels, im_test_x, im_test_keys, im_test_labels, normalize=True
          ):

          episode_dict = generate_episodes.read_in_episodes(episode_fn)
          np.random.seed(rnd_seed)
          episode_numbers = np.arange(1, len(episode_dict)+1)
          np.random.shuffle(episode_numbers)
          correct = 0
          total = 0
          
          saver = tf.train.Saver()
          with tf.Session() as sesh:
            saver.restore(sesh, model_fn)

            for episode in tqdm(episode_numbers, desc=f'\t{0}-shot multimodal matching tests on {len(episode_numbers)} episodes for random seed {rnd_seed:2.0f}', ncols=COL_LENGTH):
            
              episode_num = str(episode)
              # Get query iterator
              query = episode_dict[episode_num]["query"]
              query_data, query_keys, query_lab = generate_episodes.episode_data(query["keys"], sp_test_x, sp_test_keys, sp_test_labels)
              query_iterator = batching_library.speech_iterator(query_data, len(query_data), shuffle_batches_every_epoch=False)

              matching_set = episode_dict[episode_num]["matching_set"]
              M_data, M_keys, M_lab = generate_episodes.episode_data(matching_set["keys"], im_test_x, im_test_keys, im_test_labels)
              M_image_iterator = batching_library.unflattened_image_iterator(M_data, len(M_data), shuffle_batches_every_epoch=False)
              
              for feats, lengths in query_iterator: lat = sesh.run([speech_latent], feed_dict={speech_X: feats, speech_X_lengths: lengths, train_flag: False})[0]
              for feats in M_image_iterator: S_lat = sesh.run([image_latent], feed_dict={image_X: feats, train_flag: False})[0]

              if normalize:
                query_latents = (lat - lat.mean(axis=0))/lat.std(axis=0)
                matching_latents = (S_lat - S_lat.mean(axis=0))/S_lat.std(axis=0)
              else: 
                query_latents = lat
                s_latents = S_lat

              
              query_latent_labels = [query_lab[i] for i in query_iterator.indices]
              query_latent_keys = [query_keys[i] for i in query_iterator.indices]

              matching_latent_labels = [M_lab[i] for i in M_image_iterator.indices]
              matching_latent_keys = [M_keys[i] for i in M_image_iterator.indices]

              
              distances = cdist(query_latents, matching_latents, "cosine")
              indexes = np.argmin(distances, axis=1)
              
              label_matches = few_shot_learning_library.label_matches_grid_generation_2D(query_latent_labels, matching_latent_labels)
             
              
              for i in range(len(indexes)):
                total += 1
                if label_matches[i, indexes[i]]:
                  correct += 1
                  
          return correct/total

        log = ""
        acc = zero_shot_multimodal_task(
          lib["episode_1_list"], speech_test_x, speech_test_keys, speech_test_labels, 
          image_test_x, image_test_keys, image_test_labels, normalize=True
          )
        print(f'\tAccuracy of {0}-shot task: {acc*100:.2f}%')
        log += "Zero-shot accuracy of {} at rnd_seed of {} ".format(acc, lib["rnd_seed"])

        print("\tWriting: {}".format(lib["multimodal_zero_shot_log"]))
        with open(lib["multimodal_zero_shot_log"], "a") as write_results:
            write_results.write("\n{}: ".format(lib["model_instance"]) + log)

      if train:
        #_____________________________________________________________________________________________________________________________________
        #
        # Saving model logs
        #
        #_____________________________________________________________________________________________________________________________________

        print("\n" + "-"*PRINT_LENGTH)
        print("Writing some more model logs")
        print("-"*PRINT_LENGTH)

        results_fn = path.join(lib["output_fn"], lib["model_instance"]) + ".txt"
        print("\tWriting: {}".format(results_fn))
        with open(results_fn, "w") as write_results:
          write_results.write(train_log)
          write_results.write(log)
          write_results.close()

      model_setup_library.directory_management()