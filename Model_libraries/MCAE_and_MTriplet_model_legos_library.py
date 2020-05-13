#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
#
# This script contains the contruction of speech-image models aswell as their basic building 
# blocks.
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
import logging
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import timeit
import subprocess
from tqdm import tqdm
import model_setup_library
import model_legos_library

#_____________________________________________________________________________________________________________________________________
#
# General functions 
#
#_____________________________________________________________________________________________________________________________________

def saving_best_model(val_loss, min_val_loss, session, directory, saver, not_save_counter):

  if val_loss < min_val_loss:
      min_val_loss = val_loss
      path = saver.save(session, directory)
      save = "Saved"
      not_save_counter = 0
  else: 
      save = "  -  "
      path = "Did not save model"
      not_save_counter += 1

  return save, path, min_val_loss, not_save_counter

#_____________________________________________________________________________________________________________________________________
#
# Layer functions
#
#_____________________________________________________________________________________________________________________________________

def fully_connected_layer(input_x, train_flag, output_dim, activation, dropout_layer, lib, name, print_layer=True, reuse=False):
    reverse_activation_lib = model_setup_library.reverse_activation_lib()

    with tf.variable_scope(name, reuse=reuse):
        input_dim = input_x.get_shape().as_list()[-1]
        weight = tf.get_variable(
            "weight", [input_dim, output_dim], tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer()
            )
        bias = tf.get_variable(
            "bias", [output_dim], tf.float32, initializer=tf.random_normal_initializer()
            )
        next_layer = tf.add(tf.matmul(input_x, weight), bias)

        if activation is not None: 
            next_layer = activation(next_layer)
            act = "/" + reverse_activation_lib[activation]
        else: act = ""
        
        if dropout_layer:
            input_x = dropout(input_x, train_flag, keep_prob=lib["keep_prob"], noise_shape=None)
            if lib["keep_prob"] != 1.0: dropout_tag = "/dropout(" + str(lib["keep_prob"]) + ")"
            else: dropout_tag = ""
        else: dropout_tag = ""
        if print_layer: print(f'\t{"Layer type: ":<10}{"Fully connected layer":<20}\t{"Layer name: ":<10}{name+act+dropout_tag:<50}\t{"Layer size: ":<10}{next_layer.shape}')

        return next_layer

def conv_layer(input_x, train_flag, kernel, num_input_channels, filt, strides, pool_func, pool_size, activation, dropout_layer, name, lib, print_layer, reuse, **kwargs):
    reverse_activation_lib = model_setup_library.reverse_activation_lib()

    with tf.variable_scope(name, reuse=reuse):
        (length, width) = kernel
        kernel_shape = (length, width, num_input_channels, filt)
        kernel_tensor = tf.get_variable(
            "kernel_tensor", kernel_shape, 
            tf.float32, initializer=tf.glorot_uniform_initializer() 
            )
        bias_tensor = tf.get_variable(
            "bias_tensor", (filt), tf.float32, 
            initializer=tf.zeros_initializer() 
            )

        next_layer = tf.nn.conv2d(input_x, kernel_tensor, strides=[1, strides[0], strides[1], 1], padding='VALID', **kwargs)
        next_layer = tf.nn.bias_add(next_layer, bias_tensor)

        if activation is not None: 
            next_layer = activation(next_layer)
            act = "/" + reverse_activation_lib[activation]
        else: act = ""
        output_shape = [tf.shape(next_layer)[0], next_layer.get_shape()[-3], next_layer.get_shape()[-2], next_layer.get_shape()[-1]]
        if pool_size is not None: 
            next_layer = model_legos_library.pooling_layer(next_layer, pool_func, pool_size)
            pool = "/" + lib["pool_func"] + "_pool"
        else: pool = ""

        if dropout_layer:
            input_x = model_legos_library.dropout(input_x, train_flag, keep_prob=lib["keep_prob"], noise_shape=[tf.shape(input_x)[0], 1, 1, tf.shape(input_x)[3]])
            if lib["keep_prob"] != 1.0: dropout_tag = "/dropout(" + str(lib["keep_prob"]) + ")"
            else: dropout_tag = ""
        else: dropout_tag = ""

        if print_layer: print(f'\t{"Layer type: ":<10}{"CNN layer":<20}\t{"Layer name: ":<10}{name+act+pool+dropout_tag:<50}\t{"Layer size: ":<10}{next_layer.shape}')

    return next_layer, output_shape

def rnn_cell(output_dim, rnn_type, reuse, **kwargs):

    cell = tf.nn.rnn_cell.BasicRNNCell(output_dim, reuse=reuse, **kwargs)

    if rnn_type == "lstm":
        args_to_lstm = {"state_is_tuple": True}
        args_to_lstm.update(kwargs)
        cell = tf.nn.rnn_cell.LSTMCell(output_dim, reuse=reuse, **args_to_lstm)

    elif rnn_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(output_dim, reuse=reuse, **kwargs)

    return cell

def rnn_layer(input_x, train_flag, input_lengths, output_dim, name, rnn_type="lstm", keep_prob=1.0, scope=None, print_layer=True, reuse=False, **kwargs):

    testing_keep_prob = lambda: 1.0
    training_keep_prob = lambda: keep_prob
    cond_keep_prob = tf.cond(tf.equal(train_flag, tf.constant(True)),
        true_fn=training_keep_prob,
        false_fn=testing_keep_prob)

    with tf.variable_scope(name, reuse=reuse):

        cell = rnn_cell(output_dim, rnn_type, reuse, **kwargs)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=cond_keep_prob, output_keep_prob=keep_prob)
        output, final_state = tf.nn.dynamic_rnn(cell, input_x, sequence_length=input_lengths, dtype=tf.float32, scope=scope)
        if keep_prob != 1.0: dropout_tag = "/dropout(" + str(keep_prob) + ")"
        else: dropout_tag = ""
        if print_layer: 
            print(f'\t{"Layer type: ":<10}{"RNN layer":<20}\t{"Layer name: ":<10}{name+dropout_tag:<50}\t{"Layer size: ":<10}{output.shape}')

    return output, final_state

def multiple_rnn_layers(input_x, train_flag, input_lengths, hidden_layers, lib, layer_offset, print_layer=True, reuse=False, **kwargs):

    for layer_num, layer_size in enumerate(hidden_layers):
        output, final_state = rnn_layer(
          input_x, train_flag, input_lengths, layer_size, "rnn_layer_{}".format(layer_num+layer_offset), rnn_type=lib["rnn_type"], 
          keep_prob=lib["keep_prob"], print_layer=print_layer, reuse=reuse, **kwargs)
        input_x = output

    return output, final_state

#_____________________________________________________________________________________________________________________________________
#
# Model architectures
#
#_____________________________________________________________________________________________________________________________________

def fc_architecture(input_x, train_flag, enc_layers, latent_layer, dec_layers, lib, activation_lib, layer_prefix, layer_offset=0, print_layer=True, sub_network=False, reuse=False):
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    if sub_network is False: 
        input_x = tf.reshape(input_x, [-1, lib["input_dim"]])

    layer_counter = layer_offset

    for layer_num, layer_size in enumerate(enc_layers):
        input_x = fully_connected_layer(input_x, train_flag, layer_size, activation if layer_num < len(enc_layers) - 1 else None, 
            True, lib, layer_prefix + "_{}".format(layer_counter), print_layer, reuse=reuse)
        layer_counter += 1
       
    input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, layer_prefix + "_{}".format(layer_counter), print_layer, reuse=reuse)
    latent = input_x
    layer_counter += 1

    for layer_num, layer_size in enumerate(dec_layers):
        if layer_num < len(dec_layers) - 1: this_activation = activation
        elif sub_network is False: this_activation = tf.nn.softmax
        else: this_activation = None
        input_x = fully_connected_layer(input_x, train_flag, layer_size, this_activation, 
            True if layer_num < len(enc_layers) - 1 else False, lib, layer_prefix + "_{}".format(layer_counter), print_layer, reuse=reuse)
        layer_counter += 1

    output = input_x

    return {"latent": latent, "output": output}

def cnn_architecture(input_x, train_flag, enc_layers, enc_strides, pooling_lib, pool_layers, latent_layer, 
  dec_layers, dec_strides, lib, activation_lib, print_layer=True, **kwargs):

  pool_func = pooling_lib[lib["pool_func"]]
  activation = activation_lib[lib["activation"]]
  keep_prob = lib["keep_prob"]

  output_shapes = []
  output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-3], input_x.get_shape()[-2], input_x.get_shape()[-1]])

  for layer_num, layer_info in enumerate(enc_layers):
      input_x, output_shape = model_legos_library.conv_layer(
          input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], enc_strides[layer_num], pool_func, 
          pool_layers[layer_num], activation if layer_num < len(enc_layers) - 1 else None, True, "conv_layer_{}".format(layer_num+1), lib, print_layer
          )
      output_shapes.append(output_shape)
      if pool_layers[layer_num] is not None: output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-3], input_x.get_shape()[-2], input_x.get_shape()[-1]])
      
  input_x = tf.layers.flatten(input_x)
  output_shapes.append([tf.shape(input_x)[0], input_x.get_shape()[-1]])

  if lib["image_latent_func"] != "default":
      if lib["image_latent_func"] == "fc": 
          func = fc_architecture
          latent_model = func(input_x, train_flag, lib["image_latent_enc"], latent_layer, lib["image_latent_dec"], lib, activation_lib, "image_fc_layer", 0, print_layer, True)
          input_x = latent_model["output"]
          latent = latent_model["latent"]
          if output_shapes[-1][-1] not in lib["image_latent_dec"]: 
              input_x = model_legos_library.fully_connected_layer(input_x, train_flag, output_shapes.pop()[-1], None, False, lib, "image_fc_layer_{}".format(len(lib["image_latent_enc"])+len(lib["image_latent_dec"])+1), print_layer)

  else: 
      input_x = model_legos_library.fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "image_fc_layer_{}".format(0), print_layer)
      latent = input_x
      input_x = model_legos_library.fully_connected_layer(input_x, train_flag, output_shapes.pop()[-1], None, False, lib, "image_fc_layer_{}".format(1), print_layer)
      

  if input_x.shape != output_shapes[-2]: 
      input_x = tf.reshape(input_x, output_shapes.pop())

  for layer_num, layer_info in enumerate(dec_layers):
      input_x = model_legos_library.deconv_layer(
          input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], dec_strides[layer_num], 
          output_shapes.pop(), output_shapes.pop() if len(output_shapes) != 0 else None, activation if layer_num < len(dec_layers) - 1 else tf.nn.sigmoid, 
          True if layer_num < len(dec_layers) - 1 else False, "deconv_layer_{}".format(layer_num+1), lib, print_layer)
      
  output = input_x

  return {"latent": latent, "output": output}


def rnn_architecture(input_placeholders, train_flag, activation_lib, lib, output_lengths=None, print_layer=True, **kwargs):

  input_x = input_placeholders[0]
  input_lengths = input_placeholders[1]
  max_input_length = (
      tf.reduce_max(input_lengths) if output_lengths is None else
      tf.reduce_max([tf.reduce_max(input_lengths), tf.reduce_max(output_lengths)])
      )
  input_dim = input_x.get_shape().as_list()[-1]
  layer_count = 0
  enc_output, enc_final_state = model_legos_library.multiple_rnn_layers(input_x, train_flag, input_lengths, lib["speech_enc"], lib, 0, print_layer, **kwargs)
  layer_count += len(lib["speech_enc"])
  activation = activation_lib[lib["activation"]]

  if lib["rnn_type"] == "lstm":
      input_x, h = enc_final_state
  else:
      input_x = enc_final_state

  if lib["speech_latent_func"] != "default":
      if lib["speech_latent_func"] == "fc": 
          if lib["speech_dec"][0] not in lib["speech_latent_dec"]: lib["speech_latent_dec"].append(lib["speech_dec"][0])
          func = fc_architecture
          latent_model = func(input_x, train_flag, lib["speech_latent_enc"], lib["speech_latent"], lib["speech_latent_dec"], lib, activation_lib, "speech_fc_layer", 0, print_layer, True)
          input_x = latent_model["output"]
          latent = latent_model["latent"]
          fc_layer_count = len(lib["speech_latent_enc"]) + 1 + len(lib["speech_latent_dec"])

  else:
      input_x = model_legos_library.fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "speech_fc_layer_{}".format(0), print_layer)
      latent = input_x

      input_x = model_legos_library.fully_connected_layer(input_x, train_flag, lib["speech_dec"][0], activation, False, lib, "speech_fc_layer_{}".format(1), print_layer)
      fc_layer_count = 2
      
  
  layer_to_dec_dim = input_x.get_shape().as_list()[-1]
  dec_input = tf.reshape(tf.tile(input_x, [1, max_input_length]), [-1, max_input_length, layer_to_dec_dim])

  dec_output, dec_final_state = model_legos_library.multiple_rnn_layers(
      dec_input, train_flag, input_lengths if output_lengths is None else output_lengths, lib["speech_dec"], 
      lib, layer_count, print_layer, **kwargs)
  layer_count += len(lib["speech_dec"])

  dec_output = tf.reshape(dec_output, [-1, lib["speech_dec"][-1]])
  dec_output = model_legos_library.fully_connected_layer(dec_output, train_flag, input_dim, None, False, lib, "speech_fc_layer_{}".format(fc_layer_count), print_layer)
  dec_output = tf.reshape(dec_output, [-1, max_input_length, input_dim])
  
  return {"latent": latent, "output": dec_output}



def siamese_cnn_architecture(input_x, train_flag, enc_layers, enc_strides, pooling_lib, pool_layers, latent_layer, 
    lib, activation_lib, print_layer=True, reuse=False, **kwargs):

    pool_func = pooling_lib[lib["pool_func"]]
    activation = activation_lib[lib["activation"]]
    keep_prob = lib["keep_prob"]

    for layer_num, layer_info in enumerate(enc_layers):
      input_x, output_shape = conv_layer(
          input_x, train_flag, (layer_info[-3], layer_info[-2]), input_x.shape[-1], layer_info[-1], enc_strides[layer_num], pool_func, 
          pool_layers[layer_num], activation if layer_num < len(enc_layers) - 1 else None, True, "conv_layer_{}".format(layer_num+1), lib, print_layer, reuse
          )

    input_x = tf.layers.flatten(input_x)
    
    if lib["image_latent_func"] != "default":
      if lib["image_latent_func"] == "fc": 
          func = fc_architecture
          latent_model = func(input_x, train_flag, lib["image_latent_enc"], latent_layer, [], lib, activation_lib, "image_fc_layer", 0, print_layer, True, reuse)
          input_x = latent_model["output"]
          latent = input_x
          
    else: 
      input_x = fully_connected_layer(input_x, train_flag, latent_layer, None, False, lib, "image_fc_layer_{}".format(0), print_layer, reuse=reuse)
      latent = input_x

    latent = tf.nn.l2_normalize(latent, axis=1)

    return {"output": latent}

def siamese_rnn_architecture(input_placeholders, train_flag, activation_lib, lib, print_layer=True, reuse=False, **kwargs):

    input_x = input_placeholders[0]
    input_lengths = input_placeholders[1]
    
    enc_output, enc_final_state = multiple_rnn_layers(input_x, train_flag, input_lengths, lib["speech_enc"], lib, 0, print_layer, reuse=reuse, **kwargs)

    if lib["rnn_type"] == "lstm":
        input_x, h = enc_final_state
    else:
        input_x = enc_final_state

    if lib["speech_latent_func"] != "default":
        if lib["speech_latent_func"] == "fc": 
          func = fc_architecture
          latent_model = func(input_x, train_flag, lib["speech_latent_enc"], lib["speech_latent"], [], lib, activation_lib, "speech_fc_layer", 0, print_layer, reuse=reuse)
          input_x = latent_model["output"]
          latent = latent_model["latent"]

    else:   
        input_x = fully_connected_layer(input_x, train_flag, lib["latent"], None, False, lib, "speech_fc_layer_{}".format(0), print_layer, reuse=reuse)
        latent = input_x
        
    latent = tf.nn.l2_normalize(latent, axis=1)

    return {"output": latent}
