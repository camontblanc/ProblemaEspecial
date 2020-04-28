#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Carlos Montenegro, Control Research Group, Universidad de los Andes
        (GitHub: camontblanc)
"""

import os
import subprocess

from dm_control import suite

from collections import deque
from tqdm import trange

import numpy as np
import tensorflow as tf
from tensorflow import keras

from bayesian_ddpg import Agent
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)
STATE_DIM = (5,)
ACTION_DIM = 1
BATCH_SIZE = 256

env = suite.load(domain_name='cartpole', 
                 task_name='balance')

agent = Agent(state_dim=STATE_DIM, 
              action_dim=ACTION_DIM, 
              dropout_on_v=0)

print('Running on ', agent.device)

rb = ReplayBuffer(BUFFER_SIZE, {"obs": {"shape": (STATE_DIM,)},
                                "act": {"shape": ACTION_DIM},
                                "rew": {},
                                "next_obs": {"shape": (STATE_DIM,)},
                                "done": {}})

log_dir="logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "scalar/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# Checkpoint-saver
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(actor=agent.pi,
                                 critic=agent.critic,
                                 actor_optim=agent.pi_optim,
                                 critic_optim=agent.critic_optim)

cmd = 'tensorboard dev upload --logdir ' + log_dir
subprocess.call(cmd)

n_episodes=1000; max_t=1e3; save_every=2
scores_deque = deque(maxlen=save_every)


prevScore = 0
for i_episode in trange(1, int(n_episodes)+1):
    
    time_step = env.reset()
    state = np.concatenate( [ time_step.observation[key] 
                             for key in list( time_step.observation.keys() ) ] )
    score = 0
    
    for t in range(int(max_t)):      
        action = agent.get_action(state)
        time_step = env.step(action)
        reward, done = time_step.reward, time_step.last()
        next_state = np.concatenate( [ time_step.observation[key] 
                                      for key in list( time_step.observation.keys() ) ] )
        
        # Learn, if enough samples are available in memory
        if rb.get_stored_size() > BATCH_SIZE:
            data = rb.sample(BATCH_SIZE)                
            states = data['obs']; actions = data['act']; rewards = data['rew']
            next_states = data['next_obs']; dones = data['done']
            
            actor_loss, critic_loss, _ = agent.train(states, 
                                                     actions, 
                                                     next_states, 
                                                     rewards, 
                                                     dones)
            with summary_writer.as_default():
                tf.summary.scalar(name="actor_loss",
                                  data=actor_loss,
                                  step=t)
                tf.summary.scalar(name="critic_loss",
                                  data=critic_loss,
                                  step=t)
        
        # Save experience / reward
        else:       
            rb.add(obs=state, 
                   act=action, 
                   next_obs=next_state, 
                   rew=reward,
                   done=done)
            
        state = next_state
        score += reward

        if done:
            break
    
    with summary_writer.as_default():
        tf.summary.scalar(name="EpRet",
                          data=score,
                          step=i_episode)
    
    if i_episode % save_every == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
        
checkpoint.save(file_prefix = checkpoint_prefix)