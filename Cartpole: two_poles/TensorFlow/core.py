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
from tensorflow.keras.models import Sequential

tf.keras.backend.set_floatx('float64')

class Actor(Model):
    """Actor (Policy) Model."""
    def __init__(self, state_size, output_size, max_action, hidden_layers=[400, 300], name="Actor"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            output_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): Number of neurons in each layer
        """
        super(Actor, self).__init__(name=name)

        self.l1 = Dense(hidden_layers[0], name="L1")
        self.l2 = Dense(hidden_layers[1], name="L2")
        self.l3 = Dense(output_size, name="L3")
        
        self.max_action = max_action
        
    def call(self, state):
        x = tf.nn.relu(self.l1(state))
        x = tf.nn.relu(self.l2(x))
        x = self.l3(x)
        
        return tf.nn.tanh(x)
    
class Critic(Model):
    """Critic (Value) Model."""
    def __init__(self, state_size, output_size, hidden_layers=[400, 300], name="Critic"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            output_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__(name=name)
        
        self.l1 = Dense(hidden_layers[0], name="L1")
        self.l2 = Dense(hidden_layers[1], name="L2")
        self.l3 = Dense(output_size, name="L2")
        
        self.l4 = Dense(hidden_layers[0], name="L4")
        self.l5 = Dense(hidden_layers[1], name="L5")
        self.l6 = Dense(output_size, name="L6")
                
        
    def call(self, states, actions):
        xu = tf.concat([states, actions], axis=1)
        
        x1 = tf.nn.relu(self.l1(xu))
        x1 = tf.nn.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        x2 = tf.nn.relu(self.l4(xu))
        x2 = tf.nn.relu(self.l5(x2))
        x2 = self.l6(x2)
        
        return x1, x2