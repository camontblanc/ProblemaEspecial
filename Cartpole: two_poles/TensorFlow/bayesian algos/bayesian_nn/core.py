#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Carlos Montenegro, Control Research Group, Universidad de los Andes
        (GitHub: camontblanc)
"""

import numpy as np
import scipy.signal

import tensorflow as tf

import tensorflow.keras.layers as kl
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

import baselines.tf_util as U

def generate_dropout_layer(apply_layer, 
                           prev_dropout_layer, 
                           keep_prob):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        dropout_network, mask = U.bayes_dropout(dropout_network, 
                                                keep_prob)
        new_networks.append(dropout_network)
    return new_networks

def apply_to_layer(apply_layer, 
                   prev_dropout_layer):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        new_networks.append(dropout_network)
    return new_networks


class Actor(Model):
    """Actor (Policy) Model."""
    def __init__(self,
                 state_shape,
                 output_size, 
                 max_action, 
                 hidden_layers=(400, 300),
                 name="Actor", 
                 layer_norm=True):

        super(Actor, self).__init__(name=name)
        
        self.hidden_sizes = hidden_layers
        self.output_layer = Dense(output_size, name="OutLayer")
                
        self.max_action = max_action
        self._norm = layer_norm
        
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=state_shape+(1,), 
                                      dtype=np.float32)))
        
        print('\rActor LayerNorm: {}'.format(self._norm))
        
    def call(self, obs):
        x = obs
        
        for i, hid_size in enumerate(self.hidden_sizes):
            hid_layer = Dense(hid_size, 
                              name="actor_layer%i"%(i+1))
            norm_layer = kl.LayerNormalization(center=True,
                                               scale=True,
                                               name="actor_layer_norm%i"%(i+1))
            
            x = hid_layer(x)
            if self._norm:
                x = norm_layer(x)
            x = tf.nn.relu(x)
        
        x = self.output_layer(x)
        return tf.nn.tanh(x)
    
class Critic(Model):
    """Critic (Value) Model."""
    def __init__(self,
                 state_shape,
                 output_size,
                 hidden_layers=(400, 300), 
                 merge_layer=1, 
                 name = 'critic', 
                 layer_norm=True,
                 mc_samples=50, 
                 keep_prob=0.95,
                 dropout_on_v=None):
        super(Critic, self).__init__(name=name)
        
        self.output_size = output_size
        self._norm = layer_norm
        self.hidden_sizes = hidden_layers
        self.mc_samples = mc_samples
        self.keep_prob = keep_prob
        self.merge_layer = merge_layer
        self.dropout_on_v = dropout_on_v
                
        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, 
                     dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, output_size], 
                     dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)
        
        print('\rCritic LayerNorm {}. Dropout on V: {}'.format(self._norm, self.dropout_on_v))
            
        
    def call(self, obs, action):
        x = obs
        
        dropout_networks = [x] * self.mc_samples
        # dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, self.V_keep_prob)
        
        for i, hid_size in enumerate(self.hidden_sizes):
            if  i == self.merge_layer:
                x = tf.concat([x, action],
                              axis=1)
                
                apply_layer = lambda y : tf.concat([y, action], 
                                                   axis=1)
                dropout_networks = apply_to_layer(apply_layer, 
                                                  dropout_networks)
                
                
            if self._norm:
                hid_layer = Dense(hid_size, 
                                  name="critic_layer%i"%(i+1))
                norm_layer = kl.LayerNormalization(center=True,
                                                   scale=True,
                                                   name="critic_layer_norm%i"%(i+1))
                x = tf.nn.relu(
                            norm_layer( hid_layer(x) )
                )
                
                apply_layer = lambda y : tf.nn.relu(
                                                norm_layer( hid_layer(y) ) 
                )
                
            else:
                hid_layer = Dense(hid_size, 
                                  name="critic_layer%i"%(i+1))

                x = tf.nn.relu(
                            hid_layer(x)
                )
                
                apply_layer = lambda y : tf.nn.relu(
                                                hid_layer(y)
                )

            dropout_networks = generate_dropout_layer(apply_layer, 
                                                      dropout_networks, 
                                                      self.keep_prob)
             
        ## final layer
        out_layer = Dense(self.output_size,
                          name="critic_output",
                          activation=None, 
                          kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                           maxval=3e-3) 
                         )
        x = out_layer(x)
                          
        apply_layer = lambda y : out_layer(y)
        
        dropout_networks = apply_to_layer(apply_layer,
                                          dropout_networks)
        '''
        if self.dropout_on_v is not None:
            return x, tf.math.add_n( dropout_networks ) / float( len(dropout_networks) ), dropout_networks
        else:
            return x
        '''
        return x