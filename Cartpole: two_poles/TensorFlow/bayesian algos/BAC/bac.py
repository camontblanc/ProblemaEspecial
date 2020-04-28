#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Carlos Montenegro, Control Research Group, Universidad de los Andes
        (GitHub: camontblanc)
        
@article{ghavamzadeh2016bayesian,
author = {Ghavamzadeh, Mohammad and Engel, Yaakov and Valko, Michal},
journal = {Journal of Machine Learning Research},
title = {{Bayesian policy gradient and actor-critic algorithms}},
year = {2016}
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from policies.gaussian_actor import GaussianActor
    
class BAC():
    def __init__(
            self,
            state_shape,
            action_dim,
            actor=None,
            actor_units=[256, 256],
            lr_actor=1e-3,
            fix_std=False,
            const_std=0.3,
            hidden_activation_actor="relu",
            name="BayesianActorCritic",
            **kwargs):
        
        self.pi = GaussianActor(
                        state_shape, action_dim, actor_units,
                        hidden_activation=hidden_activation_actor,
                        fix_std=fix_std, const_std=const_std, state_independent_std=True)
        
        self.lr_actor = lr_actor
        
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]
        
        gpu = tf.config.experimental.list_logical_devices('GPU')
        self.device = '/CPU:0' if len(gpu)==0 else gpu[0].name
        
    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray), \
            "Input instance should be np.ndarray, not {}".format(type(state))
        
        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp, _ = self._get_action_body(state, test)

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()
        
    @tf.function
    def _get_action_body(self, state, test):
        return self.pi(state, test)
    
    def grad(self, state, test=False):
        assert isinstance(state, np.ndarray), \
            "Input instance should be np.ndarray, not {}".format(type(state))
        
        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
            
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                action, _, _ = self._get_action_body(state, test)
                log_prob = self.pi.compute_log_probs(state, action)
            score = tape.gradient(
                    log_prob, self.pi.trainable_variables)
        return score