# Change the path to load gym lib
import sys
import os
sys.path.append('./gym')
import gym
import numpy as np
import random
import math
from random import Random
env = gym.make('CartPole-v0')
rand = Random()

# Set debug mode
# print observation, done conditions, train verbose, play verbose 
DEBUG_FLAG = '0011'


X_MAX, X_DOT_MAX, THETA_MAX, THETA_DOT_MAX = env.observation_space.high
X_MIN, X_DOT_MIN, THETA_MIN, THETA_DOT_MIN = env.observation_space.low

print(env.observation_space.high)
print(env.observation_space.low)

X_SPACE_SIZE = 2
X_DOT_SPACE_SIZE = 2
THETA_SPACE_SIZE = 2
THETA_DOT_SPACE_SIZE = 2

LEARNING_RATE = 0.5
GAMMA = 0.9
EPSILON = 0.6
TRAIN_EPISODES = 300
MAX_TIMESTEPS = 200

v_table = [[[[ 2*[0.0] 
            for _ in range(THETA_DOT_SPACE_SIZE) ] 
            for _ in range(THETA_SPACE_SIZE) ]
            for _ in range(X_DOT_SPACE_SIZE) ]
            for _ in range(X_SPACE_SIZE) ]

def getByTuple(l, t):
  return l[t[0]][t[1]][t[2]][t[3]][t[4]]

def setByTuple(l, t, n):
  l[t[0]][t[1]][t[2]][t[3]][t[4]] = n

def updateVtable(newState, lastState, action, reward):
  newState_q_list = []
  for act in [0, 1]: newState_q_list.append(getByTuple(v_table, newState + (act,)))
  new_value = (1-LEARNING_RATE)*getByTuple(v_table, lastState+(action,)) + LEARNING_RATE*(reward+GAMMA*max(newState_q_list))
  setByTuple(v_table, lastState + (action,), new_value)
  return None

def getState(observation):
  # Get state of x
  if observation[0] < 0: x_state = 0
  else: x_state = 1

  # Get state of x dot
  if observation[1] < 0: x_dot_state = 0
  else: x_dot_state = 1

  # Get state of theta
  if observation[2] < 0: theta_state = 0
  else: theta_state = 1

  # Get state of theta dot
  if observation[3] < 0: theta_dot_state = 0
  else: theta_dot_state = 1

  return (x_state, x_dot_state, theta_state, theta_dot_state)

def getAction(state):
  if (rand.uniform(0.0, 1.0) < EPSILON):
    action = rand.randint(0, 1)
  else:
    if getByTuple(v_table, state + (0,)) > getByTuple(v_table, state + (1,)): 
      action = 0
    else: action = 1
  return action

def train(max_episodes, max_timestep):
  for i_episode in range(max_episodes):
    observation = env.reset()
    end_episode = False
    for t in range(max_timestep):
      env.render()
      old_state = getState(observation)
      action = getAction(old_state)
      observation, reward, done, info = env.step(action)
      new_state = getState(observation)
      if done: reward = 0
      updateVtable(new_state, old_state, action, reward)
      if DEBUG_FLAG[0] == '1':
        print(observation)
      if done and not end_episode:
        if DEBUG_FLAG[1] == '1':
          print()
          print('State is beyond threshold')
          print('x: %.4f, x_threshold: %.4f' % (observation[0], X_MAX / 2.0))
          print('theta: %.4f, thetathreshold: %.4f' % (observation[2], THETA_MAX / 2.0))
        end_episode = True
        if DEBUG_FLAG[2] == '1':
          print("Episode {}".format(i_episode))
          print("Episode finished after {} timesteps".format(t+1))
        break

def greedy(state):
  if getByTuple(v_table, state + (0,)) > getByTuple(v_table, state + (1,)): 
    return 0
  else:
    return 1

def play():
  observation = env.reset()
  timestep = 0
  while True:
    timestep += 1
    env.render()
    state = getState(observation)
    action = greedy(state)
    observation, reward, done, info = env.step(action)
    if done:
      if DEBUG_FLAG[3] == '1':
        print("Failed after {} timesteps".format(timestep))
      timestep = 0
      observation = env.reset()

print("Start training at %d training episodes and %d max timesteps" % (TRAIN_EPISODES, MAX_TIMESTEPS))
train(TRAIN_EPISODES, MAX_TIMESTEPS)
print("Training is completed, start executing the policy.")
play()

