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

from core import Actor, Critic

import tensorflow as tf
import tensorflow.keras.optimizers as ko
import tensorflow_addons as tfa

class TD3():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
            state_dim,
            action_dim,
            max_action,
            name="TD3",
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_units=[400, 300],
            critic_units=[400, 300],
            lr_critic=0.001,
            lr_actor=1e-4,
            tau=1e-3,
            gamma=0.99,
            wd=0.0001,
            **kwargs):
                
        # Actor Network (w/ Target Network)
        self.pi = Actor(state_dim, action_dim, max_action, actor_units)
        self.pi_targ = Actor(state_dim, action_dim, max_action, actor_units)
        self.pi_optim = ko.Adam(learning_rate=lr_actor)
        
        # Critic Networks (w/ Target Network)
        self.critic = Critic(state_dim, action_dim, critic_units)
        self.critic_target = Critic(state_dim, action_dim, critic_units)
        self.soft_update(
            self.critic_target.weights, self.critic.weights, tau=1.)
        self.critic_optimizer = tfa.optimizers.AdamW(learning_rate=lr_critic, weight_decay=wd)
        
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        
        self._actor_update_freq = actor_update_freq
        self._it = 0
        
        self.tau = tau
        self.gamma = gamma
        
        gpu = tf.config.experimental.list_logical_devices('GPU')
        self.device = '/CPU:0' if len(gpu)==0 else gpu[0].name
        
        
    #@tf.function
    def train_body(self, states, actions, next_states, rewards, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_error1, td_error2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean(td_error1**2) + tf.reduce_mean(td_error2**2)
            
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))
            
            self._it += 1
            with tf.GradientTape() as tape:
                next_actions = self.pi(states)
                actor_loss = - \
                    tf.reduce_mean(self.critic(states, next_actions))
                
            if self._it % self._actor_update_freq == 0:
                actor_grad = tape.gradient(
                    actor_loss, self.pi.trainable_variables)
                self.pi_optim.apply_gradients(
                    zip(actor_grad, self.pi.trainable_variables))
                
            # Update target networks
            self.soft_update(
                self.critic_target.weights, self.critic.weights, self.tau)
            self.soft_update(
                self.pi_targ.weights, self.pi.weights, self.tau)
        
    def compute_td_error(self, states, actions, next_states, rewards, dones):
        td_errors1, td_errors2 = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_errors1.numpy()) + np.abs(td_errors2.numpy()))
    
    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - dones
            
            # Get noisy action
            next_action = self.pi_targ(next_states)
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action),
                                 stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float64)
            next_action = tf.clip_by_value(
                next_action + noise, -self.pi_targ.max_action, self.pi_targ.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = rewards + (not_dones * self.gamma * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q1, current_Q2 = self.critic(states, actions)
            
            return target_Q - current_Q1, target_Q - current_Q2
    
                
    def act(self, state, add_noise=False):
        state = tf.cast( tf.reshape(state, [1, -1]), dtype=tf.float64 ) 
        action = self.pi(state)
        if add_noise:
            noise = tf.cast(tf.clip_by_value(
                tf.random.normal(shape=tf.shape(next_action),
                                 stddev=self._policy_noise),
                -self._noise_clip, self._noise_clip), tf.float32)
            
            action = tf.clip_by_value(
                action + noise, -self.pi_targ.max_action, self.pi_targ.max_action).numpy()
        return action.numpy()

            
    def soft_update(self, 
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