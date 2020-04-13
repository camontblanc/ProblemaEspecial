#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Carlos Montenegro, Control Research Group, Universidad de los Andes
        (GitHub: camontblanc)
"""

from dm_control import suite

import numpy as np
from collections import deque
from tqdm import trange
from IPython.display import clear_output

from td3 import TD3
import tensorflow as tf
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)
STATE_DIM = 5
ACTION_DIM = 1
BATCH_SIZE = 256

env = suite.load(domain_name='cartpole', task_name='swingup')
action_spec = env.action_spec()

agent = TD3(STATE_DIM, ACTION_DIM, max_action=action_spec.maximum)
print('Running on ', agent.device)

rb = ReplayBuffer(BUFFER_SIZE, {"obs": {"shape": (STATE_DIM,)},
                               "act": {"shape": ACTION_DIM},
                               "rew": {},
                               "next_obs": {"shape": (STATE_DIM,)},
                               "done": {}})

n_episodes=3; max_t=1e3; print_every=2
scores_deque = deque(maxlen=print_every)
scores = []

for i_episode in trange(1, int(n_episodes)+1):
    
    time_step = env.reset()
    state = np.concatenate( [ time_step.observation[key] 
                             for key in list( time_step.observation.keys() ) ] ).T
    score = 0
    
    for t in range(int(max_t)):
        
        action = agent.act(state)
        time_step = env.step(action)
        reward, done = time_step.reward, time_step.last()
        next_state = np.concatenate( [ time_step.observation[key] 
                                      for key in list( time_step.observation.keys() ) ] ).T
        
        # Learn, if enough samples are available in memory
        if rb.get_stored_size() > BATCH_SIZE:
            
            data = rb.sample(BATCH_SIZE)
            for k in data.keys():
                data[k] = data[k].astype('float64')
                
            states = data['obs']; actions = data['act']; rewards = data['rew']
            next_states = data['next_obs']; dones = data['done']
            
            agent.train_body(states, actions, next_states, rewards, dones)
        
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

    scores_deque.append(score)
    scores.append(score)

    if i_episode % print_every == 0:
        
        tf.saved_model.save(agent.pi,'checkpoint_critic')
        tf.saved_model.save(agent.pi,'checkpoint_actor')

        clear_output(True)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))