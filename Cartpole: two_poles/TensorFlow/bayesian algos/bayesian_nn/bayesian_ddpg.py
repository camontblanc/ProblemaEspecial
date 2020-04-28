#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Carlos Montenegro, Control Research Group, Universidad de los Andes
        (GitHub: camontblanc)
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque
from copy import deepcopy

from core import Actor, Critic

import tensorflow as tf
import tensorflow.keras.optimizers as ko
import tensorflow_addons as tfa

tf.summary.trace_on(graph=True)
tf.config.experimental_run_functions_eagerly(True)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 dropout_on_v=None,
                 dropout_tau_V=0.85,
                 name="BAC",
                 max_action=1.,
                 lr_actor=1e-4,
                 lr_critic=0.001,
                 actor_units=(400, 300),
                 critic_units=(400, 300),
                 sigma=0.1,
                 gamma=0.99,
                 tau=1e-3,
                 alpha=0.5,
                 **kwargs):
        
        
        # Actor Network (w/ Target Network)
        self.pi = Actor(state_dim, action_dim, max_action, actor_units)
        self.pi_targ = deepcopy(self.pi)
        self.pi_optim = ko.Adam(learning_rate=lr_actor)
        
        # Critic Networks (w/ Target Network)
        self.critic = Critic(state_shape=state_dim, 
                             output_size=action_dim, 
                             hidden_layers=critic_units, 
                             dropout_on_v=dropout_on_v, 
                             name='critic')
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = ko.Adam(learning_rate=lr_critic)
        
        # Set hyperparameters
        self.tau = tau
        self.dropout_on_v = dropout_on_v
        self.dropout_tau_V = dropout_tau_V
        self.gamma = gamma
        self.alpha = alpha
        self.max_action = max_action
        self.sigma = sigma

        '''
        gpu = tf.config.experimental.list_logical_devices('GPU')
        self.device = '/CPU:0' if len(gpu)==0 else "/GPU:0"
        '''
        self.device = '/CPU:0'
    
    def get_action(self, 
                   state, 
                   test=False, 
                   tensor=False):
        is_single_state = len(state.shape) == 1
        if not tensor:
            assert isinstance(state, np.ndarray)
        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(
            tf.constant(state), self.sigma * (1. - test),
            tf.constant(self.pi.max_action, dtype=tf.float32))
        if tensor:
            return action
        else:
            return action.numpy()[0] if is_single_state else action.numpy()
        
    
    def compute_td_error(self,
                         states,
                         actions, 
                         next_states, 
                         rewards, 
                         dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        
        _, td_errors = self._compute_td_error_body(states, 
                                                   actions, 
                                                   next_states, 
                                                   rewards, 
                                                   dones)
        return np.abs(np.ravel(td_errors.numpy()))
    
    def train(self,
              states, 
              actions, 
              next_states, 
              rewards, 
              done):
        actor_loss, critic_loss, td_errors = self._train_body(
            states, actions, next_states, rewards, done)
                
        return actor_loss, critic_loss, td_errors
    
    @tf.function
    def _train_body(self, 
                    states, 
                    actions, 
                    next_states, 
                    rewards, 
                    done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                 critic_loss, td_errors = self._compute_critic_loss(states,
                                                                    actions,
                                                                    next_states,
                                                                    rewards, 
                                                                    done)
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optim.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))
            
            with tf.GradientTape() as tape:
                next_action = self.pi(states)
                actor_loss = -tf.reduce_mean(self.critic(states, 
                                                         next_action))
            actor_grad = tape.gradient(
                actor_loss, self.pi.trainable_variables)
            self.pi_optim.apply_gradients(
                zip(actor_grad, self.pi.trainable_variables))
            
            self._soft_update(self.critic_target.weights,
                              self.critic.weights, 
                              self.tau)
            self._soft_update(self.pi_targ.weights,
                              self.pi.weights,
                              self.tau)

            
            return actor_loss, critic_loss, td_errors
    
    @tf.function
    def _compute_critic_loss(self, 
                               states, 
                               actions, 
                               next_states, 
                               rewards, 
                               dones):
        with tf.device(self.device):
            not_dones = 1. - dones
                            
            Q_targ = self.critic_target(next_states, 
                                        self.pi_targ(next_states))
            Q_targ = rewards + (not_dones * self.gamma * Q_targ)
            Q_targ = tf.stop_gradient(Q_targ)     
            current_Q = self.critic(states, actions)
            
            if self.dropout_on_v is not None:  # Eq. (10) - Dropout Inference in Bayesian Neural Networks with Alpha-divergences
                sumsq = tf.reduce_sum(tf.math.square(Q_targ - current_Q),
                                      axis = -1)
                sumsq *= (-.5 * self.alpha * self.dropout_tau_V)
                critic_loss = (-1. * self.alpha ** -1.) * tf.reduce_logsumexp(sumsq,
                                                                              axis=-1)
            else:
                critic_loss = tf.reduce_mean(tf.math.square(current_Q - Q_targ,
                                                            axis=-1))
                
        return critic_loss, Q_targ - current_Q
    
    @tf.function
    def _get_action_body(self, 
                         state, 
                         sigma, 
                         max_action):
        with tf.device(self.device):
            action = self.pi(state)
            action += tf.random.normal(shape=action.shape,
                                       mean=0., 
                                       stddev=sigma, 
                                       dtype=tf.float32)
            return tf.clip_by_value(action, -self.max_action, self.max_action)

                         
    
    def _soft_update(self,
                     target_variables,
                     source_variables,
                     tau=1.0, 
                     use_locking=False,
                     name="soft_update"):
        """
        Returns an op to update a list of target variables from source variables.
        
        The update rule is:
        `target_variable = (1 - tau) * target_variable + tau * source_variable`.
        
        :param target_variables: a list of the variables to be updated.
        :param source_variables: a list of the variables used for the update.
        :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
            with small tau representing an incremental update, and tau == 1
            representing a full update (that is, a straight copy).
        :param use_locking: use `tf.Variable.assign`'s locking option when assigning
            source variable values to target variables.
        :param name: sets the `name_scope` for this op.
        
        :raise TypeError: when tau is not a Python float
        :raise ValueError: when tau is out of range, or the source and target variables
            have different numbers or shapes.
            
        :return: An op that executes all the variable updates.
        """
        if not isinstance(tau, float):
            raise TypeError("Tau has wrong type (should be float) {}".format(tau))
        if not 0.0 < tau <= 1.0:
            raise ValueError("Invalid parameter tau {}".format(tau))
        if len(target_variables) != len(source_variables):
            raise ValueError("Number of target variables {} is not the same as "
                             "number of source variables {}".format(
                                 len(target_variables), len(source_variables)))
            
        same_shape = all(trg.get_shape() == src.get_shape() 
                         for trg, src in zip(target_variables, source_variables))
        if not same_shape:
            raise ValueError("Target variables don't have the same shape as source "
                             "variables.")
            
        def update_op(target_variable, source_variable, tau):
            if tau == 1.0:
                return target_variable.assign(source_variable, use_locking)
            else:
                return target_variable.assign(
                    tau * source_variable + (1.0 - tau) * target_variable, use_locking)
            
        # with tf.name_scope(name, values=target_variables + source_variables):
        update_ops = [update_op(target_var, source_var, tau) 
                      for target_var, source_var 
                      in zip(target_variables, source_variables)]
        return tf.group(name="update_all_variables", *update_ops)