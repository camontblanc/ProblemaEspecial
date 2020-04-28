from dm_control import suite

from packaging import version

from datetime import datetime
from collections import deque
from tqdm import trange
import matplotlib.pyplot as plt

from bayesian_ddpg import Agent
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from utils.logx import EpochLogger

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

BUFFER_SIZE = int(1e5)
STATE_DIM = (5,)
ACTION_DIM = 1
BATCH_SIZE = 256

env = suite.load(domain_name='cartpole', 
                 task_name='balance')

agent = Agent(state_dim=STATE_DIM, 
              action_dim=ACTION_DIM, 
              dropout_on_v=0)

logger_kwargs=dict()
logger = EpochLogger(**logger_kwargs)

print('Running on ', agent.device)

rb = ReplayBuffer(BUFFER_SIZE, {"obs": {"shape": (STATE_DIM,)},
                                "act": {"shape": ACTION_DIM},
                                "rew": {},
                                "next_obs": {"shape": (STATE_DIM,)},
                                "done": {}})

n_episodes=1000; max_t=1e3; print_every=5
scores_deque = deque(maxlen=print_every)

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
            
            agent.train(states, actions, next_states, rewards, dones)
        
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
    
    if i_episode % print_every == 0:
        
        # Log info about epoch
        logger.log_tabular('Episode', i_episode)
        logger.log_tabular('EpScore', score)
        logger.log_tabular('PrevScore', prevScore)
        logger.log_tabular('EpLen (current)', t)
        
        # Save models
        paths = logger.tf_simple_save(agent)
                
        prevScore = score
        logger.dump_tabular()
